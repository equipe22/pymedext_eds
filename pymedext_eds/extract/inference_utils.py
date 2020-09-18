from flair.models import SequenceTagger
import pandas as pd
import numpy as np
import os
from os.path import join
import flair
import torch
from flair.data import Sentence
from .termino import TerminologyEmbeddings
from .utils import load_conll, write_conll

def get_flair_sentence(data, text_col, add_tags):
    sentences = []
    #for each sentence
    for _, sent in data.groupby("sentence_id"):
        #get tokenized text
        text = sent[text_col].str.cat(sep=' ')
        #to flair format
        flair_sent = Sentence(text, use_tokenizer=False)
        #add tags if any
        for field in add_tags:
            for flair_tok, conll_tok in zip(flair_sent, sent.loc[:, field]):
                flair_tok.add_tag(tag_type=field, tag_value=conll_tok)

        sentences.append(flair_sent)
    return sentences

def add_flair_tag(data, sentences, tag_type):
    pred = [[tok.get_tag(tag_type) for tok in sent] for sent in sentences]
    data[tag_type] =  [t.value for sub in pred for t in sub]
    data[tag_type+"_score"] =  [t.score for sub in pred for t in sub]
    
    return data


def infer_conll(corpus_path,
                sets,
                text_col,
                tagger_path,
                tag_type,
                mini_batch_size = 5,
                embedding_storage_mode ="none",
                device = "cuda:7"):
    flair.device = torch.device(device)


    #init tagger
    print("loading model at", tagger_path)
    tagger = SequenceTagger.load(tagger_path)
    tagger.tag_type = tag_type

    #for elmo
    if hasattr(tagger.embeddings.embeddings[0], "ee"):
        tagger.embeddings.embeddings[0].ee.cuda_device = flair.device.index

    #get terminology tag if any
    add_tags = []
    for e in tagger.embeddings.embeddings:
        if hasattr(e, "field"):
            field = getattr(e, "field")
            add_tags.append(field)

    #for each dataset
    for s in sets:
        data = load_conll(corpus_path, s)

        sentences = get_flair_sentence(data, text_col, add_tags)

        #predict
        sentences = tagger.predict(sentences, verbose=True, mini_batch_size=mini_batch_size, embedding_storage_mode=embedding_storage_mode)

        #add flair tags
        data = add_flair_tag(data, sentences, tag_type)

        #reformat
        write_conll(data, corpus_path, s)

