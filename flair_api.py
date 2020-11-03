
from flair.embeddings import *
from flair.embeddings.legacy import BertTokenizer
import os
from torch.optim import SGD, Adam, AdamW, ASGD
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
import pandas as pd
from glob import glob
import flair
import torch
import yaml
import argparse

from pymedext_eds.extract.termino import TerminologyEmbeddings


print("initialized")


def run_exp(data_folder,
            columns,
            ter_params,
            lm_param, 
            optimizer_type,
            tagger_params, 
            trainer_params,
            exp_device, 
            ratio = 0
           ):
    flair.device = torch.device(exp_device)
    assert set(columns.keys()) == set(['text', 'ner']) , "Columns shoud only has 2 value, text and ner"
    #check every tsv in directory has the same number of columns
    check_cols = [(fn, len(pd.read_csv(fn, sep="\t", index_col=False, nrows=1).columns)) for fn in glob(data_folder + "*.tsv")]
    assert len(set([t[1] for t in check_cols])) == 1, check_cols

    #load headers
    headers = {v:i for i,v in enumerate(pd.read_csv(data_folder + "trainheaders.tsv", sep="\t").columns.tolist())}

    #define tag from params
    tag_type = 'ner'

    #make columns format for flair
    columns = {headers[v]:k for k,v in columns.items()}
    ter_cols = {headers[v['field']]:v['field'] for v in ter_params}
    merged_columns = {**columns, **ter_cols}
    assert len(merged_columns) == len(columns) + len(ter_cols), "Terminolgy dict cannot be the columns to be tagged"

    


    if optimizer_type == "ASGD":
        optimizer = ASGD
    elif optimizer_type == "AdamW":
        optimizer = AdamW
    else:
        optimizer = SGD

    print("Loading file withe columns:", merged_columns)
    print("TAG-type:", tag_type)

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, merged_columns,
                                  train_file='train.tsv',
                                  test_file='test.tsv',
                                  dev_file='dev.tsv')
        
    #oversampling
    if ratio > 0:
        print("oversample with ratio of 1 postive example for {}".format(ratio))
        count = 0
        oversampled_sentences = []
        for sent in corpus.train.sentences:
            if sent.get_spans("ner"):
                count += ratio
                oversampled_sentences.append(sent)

            else:
                if count >=0:
                    oversampled_sentences.append(sent)
                    count -= 1

        corpus.train.sentences = oversampled_sentences
        corpus.train.total_sentence_count = len(corpus.train.sentences)
        del oversampled_sentences


    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print("Tagging objective:", tag_dictionary.idx2item)

    lm_type = lm_param["lm_type"]
    if lm_type == "bert":
        if "lm_path" in lm_param:
            lm_path = lm_param["lm_path"]
        else:
            lm_path = 'data/embeddings/bert-base-medical-huge-cased/checkpoint-800000/'
            
        bert_embedding = TransformerWordEmbeddings(lm_path)
        
        if "lm_tokenizer_path" in lm_param:
            tok_path = lm_param['lm_tokenizer_path']
        else: 
            tok_path = 'data/embeddings/bert-base-multilingual-cased/'
        bert_embedding.tokenizer = BertTokenizer.from_pretrained(tok_path, do_lower_case=False)
        
        lm_embedding = [bert_embedding]
        
    elif lm_type == "camembert":
        if "lm_path" in lm_param:
            lm_path = lm_param["lm_path"]
        else:
            lm_path = "data/embeddings/camembert-base/"
            
        bert_embedding = TransformerWordEmbeddings(lm_path)
        
        lm_embedding = [bert_embedding]
        
    elif lm_type == "elmo":
        if "lm_path" in lm_param:
            lm_path = lm_param["lm_path"]
        else:
            lm_path = '../ivan_nested_NER/data/embeddings/nck_elmo/'
        lm_embedding = [ELMoEmbeddings(options_file=os.path.join(lm_path, "options.json"), weight_file= os.path.join(lm_path, "weights.hdf5"))]
    else:
        lm_embedding = []

    terminology_embeddings = [TerminologyEmbeddings(corpus=corpus, **t) for t in ter_params]

#     if terminology_embeddings:
    embeddings: StackedEmbeddings = StackedEmbeddings([ *lm_embedding, *terminology_embeddings])
#     else:
#         embeddings = lm_embedding[0]


    tagger_params = {**tagger_params,
                     **{"embeddings":embeddings,
                        "tag_dictionary":tag_dictionary,
                        "tag_type":tag_type}
                    }

    # 5. initialize sequence tagger


    tagger: SequenceTagger = SequenceTagger(**tagger_params)

    # 6. initialize trainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=optimizer)

    trainer.train(**trainer_params)
    
    return tagger, trainer



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training flair Sequence tagger')
    parser.add_argument('--config', type=str, help='Path to experiment config.')
    paras = parser.parse_args()
    config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)
    
    tagger, trainer = run_exp(**config)
