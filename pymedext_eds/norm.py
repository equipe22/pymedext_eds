from ray import serve
import requests
from typing import List

import torch

import json
import datetime

import ray
from ray.serve.utils import _get_logger
logger = _get_logger()
from tqdm import tqdm

import re

from pymedextcore.document import Document
from pymedextcore.annotators import Annotator, Annotation


from .pyromedi.romedi import Romedi
from .constants import CLASS_NORM
from .normalisation import clean_drug, clean_class, clean_freq, clean_dose

import faiss
import pandas as pd
import numpy as np
import h5py

try:
    from quickumls import QuickUMLS
    from quickumls.constants import ACCEPTED_SEMTYPES
except:
    print('QuickUMLS not installed. Please use "pip install quickumls"')


    

class NERNormalizer(Annotator):
    def __init__(self, key_input, key_output, ID, romedi_path, class_norm = CLASS_NORM): 
        
        super().__init__(key_input, key_output, ID)
        
        self.romedi = Romedi(from_cache=romedi_path)
        self.class_norm = class_norm
                
        self.cache_mentions = {"ENT/DRUG":{},
                               "ENT/CLASS":{},
                               "ENT/FREQ":{},
                               "ENT/DOSE":{}
                              }
    
    

    def normalize(self, ent_type, mention): 
        if mention.lower() in self.cache_mentions[ent_type].keys(): 
            return self.cache_mentions[ent_type][mention.lower()]
        else: 
            if ent_type == 'ENT/DRUG':
                cleaned = clean_drug(mention, self.romedi)
            elif ent_type == 'ENT/CLASS':
                cleaned = clean_class(mention, self.class_norm)
            elif ent_type == 'ENT/FREQ': 
                cleaned = clean_freq(mention)
            elif ent_type == 'ENT/DOSE': 
                cleaned = clean_dose(mention)
            else:
                raise NotImplmentedError

            self.cache_mentions[ent_type][mention.lower()] = cleaned
            return cleaned
         
    
    def annotate_function(self, _input):
        
        inps = self.get_all_key_input(_input)
        
        for annot in inps:
            annot.attributes[self.key_output] = self.normalize(annot.type, annot.value)
            
            if 'ENT/DOSE' in annot.attributes.keys() or 'ENT/FREQ' in annot.attributes.keys():
                for k,v in annot.attributes.items(): 
                    if k == 'ENT/DOSE':
                        for att in v:
                            att[self.key_output] = self.normalize('ENT/DOSE', att['value'])
                    elif k == 'ENT/FREQ':
                        for att in v:
                            att[self.key_output] = self.normalize('ENT/FREQ', att['value'])

                            
class NormPheno(Annotator):
    
    def __init__(self, key_input, key_output, ID, path_dict, k_neighboors = 5, max_n_cui = 50): 
        super().__init__(key_input, key_output, ID)
        
        self.k_neighboors = k_neighboors
        self.path_dict = path_dict
        
        #load embeddings
        hf = h5py.File(self.path_dict , 'r')

        self.dict_label = {}
        embeddings = []
        count = 0
        for cui in hf.keys():
            for label in hf[cui].keys():
                for i, vec in enumerate(hf[cui][label].values()):
                    if i > max_n_cui:
                        break
                    #todo clean
                    vec = np.array(vec)
                    if vec.ndim == 1:
                        vec = vec.reshape(1, -1)
                    if vec.shape[1] != 3072:
                        continue
                    ##
                    self.dict_label[count] = (cui, label)

                    
                    embeddings.append(vec)
                    count +=1

        assert embeddings[0].shape[0] == 1
        embeddings = np.ascontiguousarray(np.concatenate(embeddings, 0))
        

        #to faiss
        self.index_embedding = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        self.index_embedding.add(embeddings)

        
    def annotate_function(self, _input):
        
        #get annotations
        ner_annotations = self.get_all_key_input(_input)
        res = []
        
        #get embeddings
        embedding_list_annotation=[]
        for annotation in ner_annotations:
            embedding_list_annotation.append(self.get_embedding(annotation).reshape((1,-1)))
        matrix_list_annotation=np.concatenate(embedding_list_annotation)
        
        #find nearest
        nearest_neighbors = self.find_closest_embeddings(matrix_list_annotation, k = self.k_neighboors )
        
        for ann_index, annotation in enumerate(ner_annotations):  
            #get best match
            weights = nearest_neighbors[ann_index][:,1]
            indexs = nearest_neighbors[ann_index][:,0].astype('int')
            scores = np.bincount(indexs, weights = weights) / (1e-8 + np.bincount(indexs))
            emb_index = np.argmax(scores)
            distance = np.max(scores)
            cui, cui_label = self.dict_label[emb_index]

            res.append(Annotation(
                type = "normalized_mention",
                value = cui_label ,
                span = annotation.span,
                source = self.ID,
                source_ID = annotation.source_ID,
                attributes = {'score_cos':distance, "mention" : annotation.value, 'cui':cui, 'label': cui_label, **annotation.attributes})
                      )
        
        return res        

    
    def get_embedding(self, annotation):
        return annotation.attributes.pop('embedding')
    
    
    def find_closest_embeddings(self, annotation, k):
        faiss.normalize_L2(annotation)
        D, I = self.index_embedding.search(annotation, k) # sanity check
        
        return np.dstack((I,D))
    
    
class FeedDictionnary(Annotator):
    def __init__(self, 
                 key_input, 
                 key_output,
                 ID, path_dict,
                 quickumls_fp = 'data/umls2_UL/',
                 overlapping_criteria = "length", # "score" or "length"
                 threshold = 1,
                 similarity_name = "jaccard", # Choose between "dice", "jaccard", "cosine", or "overlap".
                 window = 5,
                 accepted_semtypes = {'T184', 'T047', 'T191', 'T049', 'T048', 'T046', 'T190', 'T019', 'T020', 'T037','T033'},
                 first_match_only = True,
                 max_obs_per_cui = 50
                ):
        super().__init__(key_input, key_output, ID)

        #connect and create the saving file for the embedding dictionnary
        self.path_dict = path_dict

        #init quickumls
        self.matcher = QuickUMLS(quickumls_fp= quickumls_fp,
                                 overlapping_criteria= overlapping_criteria,
                                 threshold= threshold,
                                 window= window,
                                 similarity_name= similarity_name,
                                 accepted_semtypes= accepted_semtypes)
        
        self.first_match_only = first_match_only
        self.max_obs_per_cui = max_obs_per_cui
    
    def annotate_function(self, _input):
        hf = h5py.File(self.path_dict, 'a')

        ner_annotations = self.get_all_key_input(_input)

        for annotation in ner_annotations:
            matchs = self.match(annotation.value)
            if not matchs:
                continue
                
            for match in matchs[0]:
                #force exact match
                if (match['end'] - match['start']) != len(annotation.value):
                    continue
                    
                #get embedding
                emb = annotation.attributes['embedding']
                if emb is None:
                    continue
                    
                #get group key
                groupname = match['cui'] +"/" + match['term']
                if groupname in hf:
                    group = hf.get(groupname)
                else:
                    group = hf.create_group(groupname)
                
                #save if max number of emb not reach
                if len(group) > self.max_obs_per_cui:
                    continue

                group.create_dataset(annotation.ID, data = emb)

                #save only first cui match
                if self.first_match_only:
                    break
                    
        hf.close()

        
    def match(self, text):
        return self.matcher.match(text)