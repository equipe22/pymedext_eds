import flair
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
from typing import List, Union
#from flair.datasets import SentenceDataset
#from torch.utils.data import DataLoader
from flair.training_utils import store_embeddings
from torch.utils.data import Dataset
#from flair.embeddings import BertTokenizer

#from .preprocess import TextPreprocessing
from .inference_utils import get_flair_sentence, add_flair_tag
#from .metrics_utils import conll_2_brat, brat_to_relations, get_relations
#from .termino import TerminologyEmbeddings
#from .preprocess import bert_clean_text


class Extractor(object):
    def __init__(self,  models_param, mini_batch_size, device = "cuda:7"):
        
        self.text_col = "tok"
        self.mini_batch_size = mini_batch_size
        self.embedding_storage_mode = "none"
        
        flair.device = torch.device(device)

        self.model_zoo = []
        self.tagged_name = []
        for i, param in enumerate(models_param):
#             model_state = torch.load(param["tagger_path"], map_location=flair.device)
#             tagger = SequenceTagger._init_model_with_state_dict(model_state).to(flair.device)
            
            tagger = SequenceTagger.load(param["tagger_path"]).to(flair.device)
            tagger.tag_type = param["tag_name"]
            self.tagged_name.append(param["tag_name"])
            #for elmo
            if hasattr(tagger.embeddings.embeddings[0], "ee"):
                tagger.embeddings.embeddings[0].ee.cuda_device = flair.device.index
            
            #get terminology tag if any
            add_tags = []
            for e in tagger.embeddings.embeddings:
                if hasattr(e, "field"):
                    field = getattr(e, "field")
                    add_tags.append(field)
            
            
            
            self.model_zoo.append(
                (param["tag_name"], tagger, add_tags ) 
            )
            

    def infer_flair(self, batch, flair_sentence, original_order_index):
        #for each mini_batch
        for sent_i in range(0, len(flair_sentence), self.mini_batch_size):
            mini_batch = flair_sentence[sent_i:sent_i+self.mini_batch_size]
            
            #infer each model
            for tag_name, model, add_tags in self.model_zoo:
                mini_batch = self.infer_minibatch(model, mini_batch)

                #clear embeddings
                store_embeddings(mini_batch, 'gpu')
            for sentence in mini_batch:
                sentence.clear_embeddings()
                
            flair_sentence[sent_i:sent_i+self.mini_batch_size] = mini_batch
                
        #reorder sentence
        flair_sentence = [
            flair_sentence[index] for index in original_order_index
        ]
        
        #reformat from flair to conll
        sent_count = 0
        added_tags  = [t[0] for t in self.model_zoo]
        for doc in batch:
            if doc['preprocessing_status']:
                result = flair_sentence[sent_count: sent_count+doc['n_sent']]
                sent_count+=doc['n_sent']
                assert len(result) == len(doc["flair"]), "Tokenization problem?"
                for tag in added_tags:
                    add_flair_tag(doc['conll'], result, tag_type=tag)
                    
                del doc['flair']
                doc["inference_status"] = True
            else:
                doc['inference_status'] = False
                            
        return batch

    
    def infer_minibatch(self, model, mini_batch):
        with torch.no_grad():
            if model.use_crf:
                transitions = model.transitions.detach().cpu().numpy()
            else:
                transitions = None

            feature: torch.Tensor = model.forward(mini_batch)
            tags, all_tags = model._obtain_labels(
                    feature=feature,
                    batch_sentences=mini_batch,
                    transitions=transitions,
                    get_all_tags=False,
                )

            for (sentence, sent_tags) in zip(mini_batch, tags):
                for (token, tag) in zip(sentence.tokens, sent_tags):
                    token.add_tag_label(model.tag_type, tag)
                
        return mini_batch
                            
    
    def post_process(self, conll_text):
        #tobrat
        if (conll_text.loc[:, self.tagged_name] != "O").any().any():
            all_tags = post_process_conll(conll_text)
        else:
            all_tags = []
        
        return all_tags
        

        
class ToFlair(object):
    def __init__(self, text_col, add_tags):
        self.text_col = text_col
        self.add_tags = add_tags
        
    def __call__(self, conll_text):
        flair_sentence = get_flair_sentence(conll_text, self.text_col, self.add_tags)
        
        return flair_sentence

class TextDataset(Dataset):
    def __init__(self, observations, to_flair, transform=None):
        self.transform = transform
        self.to_flair = to_flair
        self.observations = observations

    def __getitem__(self, index):
        # Load label
        observation = self.observations[index]

        try:
            assert self.transform and (observation["observation_blob"]) and (observation["observation_blob"].strip() != "")
            observation["conll"], observation["observation_blob"] = self.transform.transform_text(observation["observation_blob"])
            observation["flair"] = self.to_flair(observation["conll"])
            observation['n_sent'] = len(observation["flair"])
            observation['preprocessing_status'] = True
        except:
            observation['preprocessing_status'] = False

        return observation    
    
    def __len__(self):
        return len(self.observations) -1


    

def collate_text(batch):
    #unlist sentences
    sentences = [sent for doc in batch if doc["preprocessing_status"] for sent in doc["flair"] ]

    # reverse sort all sequences by their length
    rev_order_len_index = sorted(
        range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
    )
    original_order_index = sorted(
        range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
    )

    reordered_sentences: List[Union[Sentence, str]] = [
        sentences[index] for index in rev_order_len_index
    ]
    
    
    return batch, reordered_sentences, original_order_index