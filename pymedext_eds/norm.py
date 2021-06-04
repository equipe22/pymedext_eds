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
    
import edlib
import heapq
from operator import itemgetter
import pickle

from dtaidistance import dtw_ndim

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
                type = self.key_output,
                value = cui_label ,
                span = annotation.span,
                source = self.ID,
                source_ID = annotation.source_ID,
                attributes = {'score_cos':distance, "mention" : annotation.value, 'cui':cui, 'label': cui_label, **annotation.attributes})
                      )
        
        return res        

    
    def get_embedding(self, annotation):
        return annotation.attributes['embedding']
    
    
    def find_closest_embeddings(self, annotation, k):
        faiss.normalize_L2(annotation)
        D, I = self.index_embedding.search(annotation, k) # sanity check
        
        return np.dstack((I,D))
    
    
class NormPhenoLev(Annotator):
    
    def __init__(self, 
                 key_input,
                 key_output,
                 ID, path_dict = None,
                 make_code = False,
                 best_code_size = 4,
                 pca_dim = 128,
                 max_n_cui = 50,
                 max_editdistance = 0.5,
                 code_base = 8,
                 emb_dim = 3072
                 
                ): 
        super().__init__(key_input, key_output, ID)
        
        self.pca_dim = pca_dim
        self.path_dict = path_dict
        self.max_editdistance = max_editdistance
        self.emb_dim = emb_dim
        
        if make_code:
            #load embeddings
            hf = h5py.File(self.path_dict , 'r')

            self.dict_label = {}
            embeddings = []
            count = 0
            for cui in tqdm(hf.keys()):
                for label in hf[cui].keys():
                    for ent_i, vec in enumerate(hf[cui][label].values()):
                        vec = np.array(vec)
                        if vec.shape[1] != emb_dim:
                            continue
                        self.dict_label[count] = {"cui":cui, "label":label, "n_tok":vec.shape[0]}

                        #iterate tokens
                        for tok_i in range(len(vec)):
                            embeddings.append(vec[tok_i,:])
                            count +=1

            assert len(embeddings) == sum([t["n_tok"] for t in self.dict_label.values()])

            embeddings = np.ascontiguousarray(np.array(embeddings)).astype('float32')
            
            #fit PCA
            print("Fit PCA")
            self.pca = faiss.PCAMatrix(self.emb_dim, self.pca_dim)
            self.pca.train(embeddings)
            assert self.pca.is_trained
            embeddings = self.pca.apply_py(embeddings)
            #save PCA
            faiss.write_VectorTransform(self.pca, self.path_dict.split('.')[0]+".pca")


            print("Fit PQ")
            cs_search = {}
            for code_size in [1, 2, 4, 8, 16, 32]:
                if code_size // self.pca_dim > 0.5:
                    continue
                pq = faiss.ProductQuantizer(self.pca_dim, code_size, code_base)
                pq.train(embeddings)
                # encode 
                codes = pq.compute_codes(embeddings)
                # decode
                embeddings2 = pq.decode(codes)
                # compute reconstruction error
                avg_relative_error = ((embeddings - embeddings2)**2).sum() / (embeddings ** 2).sum()

                cs_search[code_size] = avg_relative_error
                print("code size:", code_size, "average error:", avg_relative_error)


            print("Choosing code size:", best_code_size)
            pq = faiss.ProductQuantizer(self.pca_dim, best_code_size, code_base)
            pq.train(embeddings)
            # encode 
            codes = pq.compute_codes(embeddings)

            for ent_k in self.dict_label.keys():
                self.dict_label[ent_k]['seq'] = codes[ent_k:ent_k+self.dict_label[ent_k]['n_tok']].reshape(-1).tolist()

            #self.dict_label = {i:v for i,v in enumerate(self.dict_label.values())}
            #drop exact same sequence
            new_dict = {}
            for k,v in self.dict_label.items():
                cui = v['cui']
                if cui in new_dict:
                    if v['seq'] in [t['seq'] for t in new_dict[cui]]:
                        pass
                    else:
                        new_dict[cui].append({'label':v['label'], 'seq':v['seq'], 'cui':v['cui']})

                else:
                    new_dict[cui] = [{'label':v['label'], 'seq':v['seq'], 'cui':v['cui']}]

            new_dict = [t for k, sub in new_dict.items() for t in sub]
            new_dict = {i:v for i,v in enumerate(new_dict)}
            self.dict_label = new_dict
            del new_dict
            
            self.pq = pq
            
            faiss.write_ProductQuantizer(pq, self.path_dict.split('.')[0]+".pq")


            with open(self.path_dict.split('.')[0]+".pickle", 'wb') as handle:
                pickle.dump(self.dict_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            with open(self.path_dict.split('.')[0]+".pickle", 'rb') as handle:
                self.dict_label = pickle.load(handle)

            self.pca = faiss.read_VectorTransform(self.path_dict.split('.')[0]+".pca")
            self.pq = faiss.read_ProductQuantizer(self.path_dict.split('.')[0]+".pq")




        
    def annotate_function(self, _input):
        
        #get annotations
        ner_annotations = self.get_all_key_input(_input)
        res = []
        

        for ann_index, annotation in enumerate(ner_annotations):
            #get embeddings
            emb = self.get_embedding(annotation)

            #get codes
            emb = self.get_codes(emb).reshape(-1).tolist()
            
            #get closest match
            best_match = self.get_closest_match(emb)
            if best_match["score_ed"] >= 1:
                continue
                
            
            res.append(Annotation(
                type = self.key_output,
                value = best_match["label"],
                span = annotation.span,
                source = self.ID,
                source_ID = annotation.source_ID,
                attributes = {'score_ed':best_match["score_ed"], "mention" : annotation.value, 'cui':best_match["cui"], 'label': best_match["label"], **annotation.attributes})
                      )
        
        return res        

    
    def get_embedding(self, annotation):
        return annotation.attributes['embedding']
    
    
    def get_codes(self, emb):
        emb = self.pca.apply_py(emb)
        codes = self.pq.compute_codes(emb)
        return codes
    
    def get_closest_match(self, query):
        distances = []
        best_md = 1000
        #iter possible match
        for target in self.dict_label.values():
            #compute unorm max distance
            md = max(map(len, [query, target["seq"]]))
            max_editdistance = int(md * self.max_editdistance)
            #get min btw max distance and min res
            max_editdistance = min(best_md, max_editdistance)
            res = edlib.align(query = query, target = target["seq"], task ='distance', k = max_editdistance)['editDistance']
            if res == -1:
                res = md
            best_md = min(res, best_md)
            distances.append(res / md)
            
        i_val = heapq.nsmallest(1, enumerate(distances), key=itemgetter(1))
        best_match = [self.dict_label[t[0]] for t in i_val][0]
        best_match.update({"score_ed": i_val[0][1]})
        
        return best_match if i_val[0][1] <= self.max_editdistance else {"cui":None, "score_ed":1, "label":None}
            
class NormPhenoDTW(Annotator):
    
    def __init__(self, 
                 key_input,
                 key_output,
                 ID, path_dict = None,
                 make_code = False,
                 pca_dim = 128,
                 max_editdistance = 50,
                 emb_dim = 3072,
                 dtw_window = 4
                 
                ): 
        super().__init__(key_input, key_output, ID)
        
        self.pca_dim = pca_dim
        self.path_dict = path_dict
        self.max_editdistance = max_editdistance
        self.emb_dim = emb_dim
        self.dtw_window = dtw_window
        if make_code:
            #load embeddings
            hf = h5py.File(self.path_dict , 'r')

            self.dict_label = {}
            embeddings = []
            count = 0
            for cui in tqdm(hf.keys()):
                for label in hf[cui].keys():
                    for ent_i, vec in enumerate(hf[cui][label].values()):
                        vec = np.array(vec)
                        if vec.shape[1] != emb_dim:
                            continue
                        self.dict_label[count] = {"cui":cui, "label":label, "n_tok":vec.shape[0]}

                        #iterate tokens
                        for tok_i in range(len(vec)):
                            embeddings.append(vec[tok_i,:])
                            count +=1

            assert len(embeddings) == sum([t["n_tok"] for t in self.dict_label.values()])

            embeddings = np.ascontiguousarray(np.array(embeddings)).astype('float32')
            
            #fit PCA
            print("Fit PCA")
            self.pca = faiss.PCAMatrix(self.emb_dim, self.pca_dim)
            self.pca.train(embeddings)
            assert self.pca.is_trained
            embeddings = self.pca.apply_py(embeddings)
            #save PCA
            faiss.write_VectorTransform(self.pca, self.path_dict.split('.')[0]+"_dtw.pca")


            for ent_k in self.dict_label.keys():
                self.dict_label[ent_k]['seq'] = embeddings[ent_k:ent_k+self.dict_label[ent_k]['n_tok']]

            self.dict_label = {i:v for i,v in enumerate(self.dict_label.values())}
 

            with open(self.path_dict.split('.')[0]+"_dtw.pickle", 'wb') as handle:
                pickle.dump(self.dict_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            with open(self.path_dict.split('.')[0]+"_dtw.pickle", 'rb') as handle:
                self.dict_label = pickle.load(handle)

            self.pca = faiss.read_VectorTransform(self.path_dict.split('.')[0]+"_dtw.pca")




        
    def annotate_function(self, _input):
        
        #get annotations
        ner_annotations = self.get_all_key_input(_input)
        res = []
        

        for ann_index, annotation in enumerate(ner_annotations):
            #get embeddings
            emb = self.get_embedding(annotation)

            #get codes
            emb = self.get_codes(emb)
            
            #get closest match
            best_match = self.get_closest_match(emb)
                
            
            res.append(Annotation(
                type = self.key_output,
                value = best_match["label"],
                span = annotation.span,
                source = self.ID,
                source_ID = annotation.source_ID,
                attributes = {'score_ed':best_match["score_ed"], "mention" : annotation.value, 'cui':best_match["cui"], 'label': best_match["label"], **annotation.attributes})
                      )
        
        return res        

    
    def get_embedding(self, annotation):
        return annotation.attributes['embedding']
    
    
    def get_codes(self, emb):
        emb = self.pca.apply_py(emb)
        return emb
    
    def get_closest_match(self, query):
        distances = []
        best_md = 1e18
        #iter possible match
        for target in self.dict_label.values():

            max_editdistance = min(best_md, self.max_editdistance)
            res = dtw_ndim.distance(query.astype('double'), target["seq"].astype('double'), window = self.dtw_window, max_dist=max_editdistance, use_c= True)
            best_md = min(res, best_md)
            distances.append(res)
            
        i_val = heapq.nsmallest(1, enumerate(distances), key=itemgetter(1))
        best_match = [self.dict_label[t[0]] for t in i_val][0]
        best_match.update({"score_ed": i_val[0][1]})
        return best_match

        #return best_match if i_val[0][1] <= self.max_editdistance else {"cui":None, "score_ed":1000, "label":None}
    
class NormPhenoCat(Annotator):
    
    def __init__(self, 
                 key_input,
                 key_output,
                 ID, 
                 path_dict = None,
                 make_code = False,
                 pca_dim = 50,
                 k_neighboors = 1,
                 max_distance = 0.5,
                 emb_dim = 3072
                 
                ): 
        super().__init__(key_input, key_output, ID)
        
        self.pca_dim = pca_dim
        self.path_dict = path_dict
        self.max_editdistance = max_editdistance
        self.emb_dim = emb_dim
        
        self.init_classifier(make_code)
        
    def init_classifier(self, make_code):
        if make_code:
            #load embeddings
            hf = h5py.File(self.path_dict , 'r')

            self.dict_label = {}
            embeddings = []
            count = 0
            for cui in hf.keys():
                for label in hf[cui].keys():
                    for ent_i, vec in enumerate(hf[cui][label].values()):
                        vec = np.array(vec)
                        if vec.shape[1] != emb_dim:
                            continue
                        self.dict_label[count] = {"cui":cui, "label":label, "n_tok":vec.shape[0]}

                        #iterate tokens
                        for tok_i in range(len(vec)):
                            embeddings.append(vec[tok_i,:])
                            count +=1

            assert len(embeddings) == sum([t["n_tok"] for t in self.dict_label.values()])

            embeddings = np.ascontiguousarray(np.array(embeddings)).astype('float32')


            #reduce dimension PCA
            
            
            #save PCA
            self.pq = pq
            faiss.write_ProductQuantizer(pq, self.path_dict.split('.')[0]+".pq")


            
            #concat tokens

            for ent_k in self.dict_label.keys():
                self.dict_label[ent_k]['seq'] = codes[ent_k:ent_k+self.dict_label[ent_k]['n_tok']].reshape(-1).tolist()

            #drop exact close vec
            new_dict = {}
            for k,v in self.dict_label.items():
                cui = v['cui']
                if cui in new_dict:
                    if v['seq'] in [t['seq'] for t in new_dict[cui]]:
                        pass
                    else:
                        new_dict[cui].append({'label':v['label'], 'seq':v['seq'], 'cui':v['cui']})

                else:
                    new_dict[cui] = [{'label':v['label'], 'seq':v['seq'], 'cui':v['cui']}]

            new_dict = [t for k, sub in new_dict.items() for t in sub]
            new_dict = {i:v for i,v in enumerate(new_dict)}
            self.dict_label = new_dict
            del new_dict
            

            #save
            with open(self.path_dict.split('.')[0]+".pickle", 'wb') as handle:
                pickle.dump(self.dict_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            #load dic
            with open(self.path_dict.split('.')[0]+".pickle", 'rb') as handle:
                self.dict_label = pickle.load(handle)
                
                
            #load PCA

            self.pq = faiss.read_ProductQuantizer(self.path_dict.split('.')[0]+".pq")
            
            
        #make index

        #to faiss
        self.index_embedding = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        self.index_embedding.add(embeddings)





        
    def annotate_function(self, _input):
        
        #get annotations
        ner_annotations = self.get_all_key_input(_input)
        res = []
        

        for ann_index, annotation in enumerate(ner_annotations):
            #get embeddings
            emb = self.get_embedding(annotation)

            #get codes
            emb = self.get_codes(emb).reshape(-1).tolist()
            
            #get closest match
            best_match = self.get_closest_match(emb)
            if best_match["score_ed"] >= 1:
                continue
                
            
            res.append(Annotation(
                type = self.key_output,
                value = best_match["label"],
                span = annotation.span,
                source = self.ID,
                source_ID = annotation.source_ID,
                attributes = {'score_ed':best_match["score_ed"], "mention" : annotation.value, 'cui':best_match["cui"], 'label': best_match["label"], **annotation.attributes})
                      )
        
        return res        

    
    def get_embedding(self, annotation):
        return annotation.attributes['embedding']
    
    
    def get_codes(self, emb):
        codes = self.pq.compute_codes(emb)
        return codes
    
    def get_closest_match(self, query):
        distances = []
        best_md = 1000
        #iter possible match
        for target in self.dict_label.values():
            #compute unorm max distance
            md = max(map(len, [query, target["seq"]]))
            max_editdistance = int(md * self.max_editdistance)
            #get min btw max distance and min res
            max_editdistance = min(best_md, max_editdistance)
            res = edlib.align(query = query, target = target["seq"], task ='distance', k = max_editdistance)['editDistance']
            if res == -1:
                res = md
            best_md = min(res, best_md)
            distances.append(res / md)
            
        i_val = heapq.nsmallest(1, enumerate(distances), key=itemgetter(1))
        best_match = [self.dict_label[t[0]] for t in i_val][0]
        best_match.update({"score_ed": i_val[0][1]})
        
        return best_match if i_val[0][1] <= self.max_editdistance else {"cui":None, "score_ed":1, "label":None}
            
    
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