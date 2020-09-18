from torch.utils.data import DataLoader, Dataset

from .extract.extractor import  ToFlair, collate_text, Extractor
from .extract.preprocess import TextPreprocessing
from .extract.postprocess import ConllPostProcess

from ray import serve
import requests
from typing import List

import torch
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import json

import ray
from ray.serve.utils import _get_logger
logger = _get_logger()


class TextDataset(Dataset):
    def __init__(self, observations, to_flair, transform=None):
        self.transform = transform
        self.to_flair = to_flair
        self.observations = observations

    def __getitem__(self, index):
        # Load label
        observation = self.observations[index]

        try:
            assert self.transform and (observation["note_text"]) and (observation["note_text"].strip() != "")
            observation["conll"], observation["note_text"] = self.transform.transform_text(observation["note_text"])
            observation["flair"] = self.to_flair(observation["conll"])
            observation['n_sent'] = len(observation["flair"])
            observation['preprocessing_status'] = True
        except:
            observation['preprocessing_status'] = False

        return observation    
    
    def __len__(self):
        return len(self.observations) -1

class Annotator(Extractor):
    def __init__(self, params, postprocess_params):
        
        self.text_col = "tok"
        self.mini_batch_size = 32
        self.embedding_storage_mode = "none"
        
        logger.info(ray.get_gpu_ids())
        
        if ray.get_gpu_ids() != []:
            device = f'cuda:0'
        else:
            device = 'cpu'
        
        flair.device = torch.device(device)
        
        self.preprocessor = TextPreprocessing(**params['preprocessing_param'])
        self.postprocessor =  ConllPostProcess(**postprocess_params['pp_params'])

        self.model_zoo = []
        self.tagged_name = []
        for i, param in enumerate(params['models_param']):
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
            
    def preprocess(self, chunk): 
        dataset = TextDataset(observations = chunk, to_flair = ToFlair("clean_text", []), transform = self.preprocessor)
        dataloader = DataLoader(
        dataset=dataset, shuffle = False, batch_size=100, collate_fn=collate_text, num_workers = 1
                        )
        return dataloader
    
    def process(self, chunk):
        
        dataloader = self.preprocess(chunk)
        
        res = []
        for batch_i, (batch, reordered_sentences, original_order_index) in enumerate(dataloader):
            batch = self.infer_flair(batch, reordered_sentences, original_order_index)
            for b in batch:
                
                if 'conll' in b.keys():
                    b['conll']['note_id'] = b['note_id']
                    res.append(self.postprocessor.pp_conll_simple(b['conll'])[0])
                
        return res
    
         
    def postprocess(self, conll): 
        pp = self.postprocessor.pp_conll_simple(conll)[0]
        return pp
    
    def __call__(self, flask_request):
        
        payload = flask_request.json
        res = self.process(payload)
        
        return {'result':res}


class BatchAnnotator(Annotator):
    
    @serve.accept_batch
    def __call__(self, flask_requests: List):
        
        res = []
        
        for flask_request in flask_requests:
            
            payload = flask_request.json
            res.append(self.process(payload))
        
        return {'result':payload}

