from torch.utils.data import DataLoader, Dataset

from ray import serve
import requests
from typing import List

import torch
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import json
import datetime

import ray
from ray.serve.utils import _get_logger
logger = _get_logger()
from tqdm import tqdm

import re

from flair.training_utils import store_embeddings
from pymedextcore.document import Document
from pymedextcore.annotators import Annotator, Annotation
from flair.models import SequenceTagger, MultiTagger
from flair.data import Sentence


    

class NERAnnotator(Annotator):
    def __init__(self, key_input, key_output, ID, 
                 models_param,
                 mini_batch_size = 1000,
                 emb_key = '0-transformer-word-data/embeddings/bert-base-medical-huge-cased/checkpoint-800000/',
                 reduce_embedding = 'mean',
                 temperature = 1e-8,
                 device="cpu"): 
        """
        Annotate text using collections of Flair Sequence Taggers and 
        param reduce_embedding: 'mean', 'weighted_mean' or 'concat'
        """
        
        super().__init__(key_input, key_output, ID)
        
        self.store_embedding_tag = []
        self.store_embedding_labels = []
        self.emb_key = emb_key
        self.reduce_embedding = reduce_embedding
        self.temperature = temperature
        self.mini_batch_size = mini_batch_size
        flair.device = torch.device(device)
        
        self.model_zoo = []
        self.tagged_name = []
        for i, param in enumerate(models_param):     
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
                    
                    
            #reset batch sizes
            tagger.embeddings.list_embedding_0.batch_size =  mini_batch_size

            self.model_zoo.append(
                (param["tag_name"], tagger, add_tags ) 
            )
            
            if 'store_embedding' in param.keys() and param['store_embedding']:
                self.store_embedding_tag.append(param["tag_name"])
                emb_labels = [t.split('-') for t in tagger.tag_dictionary.get_items()]
                emb_labels = [t[1] for t in emb_labels if len(t)==2]
                emb_labels = list(set(emb_labels))
                self.store_embedding_labels.extend(emb_labels)
        
        self.emb_size = 3072


    def infer_flair(self, sentences):
        flair_sentences = []
        for s in sentences: 
            flair_sentences.append(Sentence(s.value,start_position=s.span[0]))
            
        flair_sentences = self._filter_empty_sentences(flair_sentences)
        
        #for each mini_batch
        for sent_i in range(0, len(flair_sentences), self.mini_batch_size):
            mini_batch = flair_sentences[sent_i:sent_i+self.mini_batch_size]
            
            #infer each model
            for tag_name, model, add_tags in self.model_zoo:
                mini_batch = self.infer_minibatch(model, mini_batch)
                
            #clear embeddings for each token
            for sentence in mini_batch:
                for token in sentence:
                    clear_embed = True
                    # except if tagged
                    for tag_type in self.store_embedding_tag:
                        if token.get_tag(tag_type).value != 'O':
                            clear_embed = False
                    if clear_embed:
                        token.clear_embeddings()
                    #keep embedding for normalisation
                    else:
                        token._embeddings = {k:v.cpu() for k,v in token._embeddings.items() if k ==  self.emb_key}


            flair_sentences[sent_i:sent_i+self.mini_batch_size] = mini_batch
            
        return flair_sentences
    
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


    
    def _get_sentence_entities(self, flair_sentence, source):
        
        res = []
        for entity in flair_sentence.get_spans():
            #todo multiple tags ??
            label = entity.get_labels()[0]
            
            if label.value in self.store_embedding_labels:
                token_list = [t._embeddings[self.emb_key].reshape(1, -1) for t in entity.tokens]
                if self.reduce_embedding == 'mean':
                    ent_embedding = torch.cat(token_list , axis = 0).mean(axis=0).numpy()
                elif self.reduce_embedding == 'weighted_mean':
                    weights = torch.tensor([t.annotation_layers['pheno_pred'][0].score for t in entity])
                    weights = ((weights + self.temperature) / (weights.sum() + len(weights)*self.temperature))
                    ent_embedding = (torch.diag(weights) @ torch.cat(token_list , axis = 0)).sum(axis=0).numpy()
                elif self.reduce_embedding == 'concat':
                    ent_embedding = torch.cat(token_list , axis = 0).numpy()
                else: 
                    raise NotImplementedError("Reduce embedding should be one of {'concat', 'mean', 'weighted_mean'}")

                
                ent_embedding = ent_embedding.reshape(-1, self.emb_size)

            else:
                ent_embedding = None
            # filter out events
            if label.value[0:3] != 'EV/':
                res.append({'value': entity.text,
                            'span':(entity.start_pos+flair_sentence.start_pos, entity.end_pos+flair_sentence.start_pos),
                            'type': label.value,
                            'attributes': {'score':label.score, 'embedding':ent_embedding },
                            'source_ID': source.ID}
                          )
              
        return res
   
    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            logger.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences
    
    def _postprocess_entities(self, sentences, flair_sentences):
        entities = []
        
        for sentence in flair_sentences: 
            if sentence.get_spans() == []:
                continue 
                
            source = [s for s in sentences if s.span[0] == sentence.start_pos][0]
            entities.append(self._get_sentence_entities(sentence, source))
            
        return entities
    
    def _tag_sentences(self,sentences): 
        
        flair_sentences = []
        for s in sentences: 
            flair_sentences.append(Sentence(s.value,start_position=s.span[0]))
            
        self.multi.predict(flair_sentences)
        
        return flair_sentences
        
        
    
    def annotate_function(self, _input):
    
        sentences = self.get_all_key_input(_input)      
        flair_sentences = self.infer_flair(sentences)
        annotated_sentences = self._postprocess_entities(sentences, flair_sentences)
        
        res = []
        
        if len(annotated_sentences) == 0:
            return res
        
        for sent in annotated_sentences:
            

            
            if len(sent) == 0:
                continue
            
            sent_att = [x.attributes.copy() if x.attributes else None for x in sentences if x.ID == sent[0]['source_ID']][0]
            sent_att = sent_att if sent_att else {}
            
            #pheno post processing
            for ent in sent:
                if ent['type'] in ['ENT/SIGNS',
                                    'ENT/DIAG_NAME',
                                    'ENT/DIAG_PROC',
                                    'ENT/THERAP_PROC', 
                                     'ENT/NOM',
                                    'ENT/PRENOM'
                                  ]:

                    res.append(Annotation(
                            type = ent['type'],
                            value = ent['value'],
                            span = ent['span'],
                            source = self.ID,
                            isEntity = True,
                            source_ID = ent['source_ID'],
                            attributes = {**sent_att, **ent['attributes']} 
                        ))

            # if only 1 entity in sent => append Annotation
            if len(sent) == 1: 
                if sent[0]['type'] in ['ENT/DRUG','ENT/CLASS']:

                    res.append(Annotation(
                        type = sent[0]['type'],
                        value = sent[0]['value'],
                        span = sent[0]['span'],
                        isEntity = True,
                        source = self.ID,
                        source_ID = sent[0]['source_ID'],
                        attributes = {**sent_att, **sent[0]['attributes']}
                    ))
                    
                    #pprint('---- drug not blob ----')
                    #pprint (sent[0])
                    
               # else:
               #     pprint('---- isolated not drug ----')
               #     pprint (sent[0])
            
            # if more than 1 entity
            else: 
                
                new_sent = []
                
                drug_blobs = [x for x in sent if x['type'] == 'REL/DRUG_BLOB']
                
                # if drug_blobs => combine all entities in drub_blob
                if len(drug_blobs) > 0:
                    
                    blob_ents = [[] for blob in drug_blobs]
                    
                    for ent in sent: 
                        ent['in_blob'] = False
                        for i, blob in enumerate(drug_blobs): 
                            blob_start = blob['span'][0]
                            blob_end = blob['span'][1]
                            if ent['span'][0] >= blob_start and ent['span'][1] <= blob_end:
                                ent['in_blob'] = True
                                blob_ents[i].append(ent)
                                
                        # if entity not in blob => append Annotation (if ENT/DRUG or ENT_CLASS)
                        if not ent['in_blob']:
                            
                            if ent['type'] in ['ENT/DRUG','ENT/CLASS']:
                    
                                res.append(Annotation(
                                    type = ent['type'],
                                    value = ent['value'],
                                    span = ent['span'],
                                    isEntity = True,
                                    source = self.ID,
                                    source_ID = ent['source_ID'],
                                    attributes = {**sent_att, **ent['attributes']}
                                )) 
                                
                               # pprint('---- drug not blob ----')
                               # pprint (ent)
                        
                           # else: 
                                
                           #     pprint('---- not blob and not drug ----')
                           #     pprint(ent)
                    
                    # in each blob combine drug and class with their attributes
                    for blob in blob_ents: 
                        drugs_class = [x for x in blob if x['type'] in ['ENT/DRUG','ENT/CLASS']]
                        att_list = [x for x in blob if x['type'] not in ['ENT/DRUG','ENT/CLASS', 'REL/DRUG_BLOB']]
                        snippet = [x['value'] for x in blob if x['type'] == 'REL/DRUG_BLOB'][0]
                    
                        att_reformat = {}
                        for att in att_list:
                            if att['type'] not in att_reformat.keys():
                                att_reformat[att['type']] = []
                            att_reformat[att['type']] += [att]

                        #combined = []

                        for drug in drugs_class:

                            drug['attributes'] = {**sent_att, **drug['attributes'], **att_reformat}
                            drug['attributes']['snippet'] = snippet

                           # combined.append(drug)
                            
                            res.append(Annotation(
                                    type = drug['type'],
                                    value = drug['value'],
                                    span = drug['span'],
                                    isEntity = True,
                                    source = self.ID,
                                    source_ID = drug['source_ID'],
                                    attributes = drug['attributes']
                                )) 


                        #pprint('---- in blob ----')
                        #pprint(combined)
                    
                    
                
                else:
                    # keep drug and class only
                    #pprint('---- other -----')
                    
                    for ent in sent:
                        if ent['type'] in ['ENT/DRUG','ENT/CLASS']:
                    
                            res.append(Annotation(
                                    type = ent['type'],
                                    value = ent['value'],
                                    span = ent['span'],
                                    isEntity = True,

                                    source = self.ID,
                                    source_ID = ent['source_ID'],
                                    attributes = {**sent_att, **ent['attributes']}
                                ))
        return res
    
    @staticmethod
    def doc_to_omop(annotated_doc, new_norm = False): 
    
        annots = [x.to_dict() for x in annotated_doc.get_annotations('ENT/DRUG') + annotated_doc.get_annotations('ENT/CLASS') ]
        sentences = [x.to_dict() for x in annotated_doc.get_annotations('sentence')]
        if annots == []:
            return []

        note_id = annotated_doc.source_ID
        person_id = annotated_doc.attributes.get('person_id')

        res = []

        for drug in annots: 

            section = drug["attributes"]['section']
            sentence = [x['value'] for x in sentences if x['ID'] == drug['source_ID']][0]

            norm = None
            if 'normalized_mention' in drug["attributes"].keys(): 
                if (drug['attributes']['normalized_mention'] != []) & (drug['attributes']['normalized_mention'] != {}):
                    if drug['type'] == 'ENT/DRUG':
                        norm = drug['attributes']['normalized_mention']['ATC7']
                    else: 
                        norm = drug['attributes']['normalized_mention']

                    # new normalization with concept code separated from label 
                    # deactivated by default to ensure retro-compatibility
                    if norm is not None and new_norm :
                        norm_extract = re.search('^([A-Z0-9]+) \((.+)\)', norm)
                        if norm_extract:
                            norm = norm_extract.group(1)
                            drug['attributes']['ATC_LABEL'] = norm_extract.group(2)
            dose = None
            dose_norm = None
            route = None
            duration = None 
            duration_norm = None
            freq = None
            freq_norm = None

            modifiers = []
            if drug['attributes'] != {}: 
                for att,val in drug['attributes'].items():

                    if att in ['ENT/ROUTE','ENT/DOSE','ENT/DURATION','ENT/FREQ','ENT/CONDITION']:
                        for v in val:
                            modifiers.append(f"{att}='{v['value']}'")
                            if 'normalized_value' in v.keys():
                                modifiers.append(f"{att}_norm='{v['normalized_mention']}'")

                    elif att != "normalized_mention":
                        modifiers.append(f"{att}='{val}'")

            note_nlp_item = {
            'note_nlp_id' : None,
            'note_id': note_id,
            'person_id': person_id,
            'section_concept_id': section,
            'snippet': sentence, 
            'offset_begin': drug['span'][0],
            'offset_end': drug['span'][1], 
            'lexical_variant': drug['value'],
            'note_nlp_concept_id': norm,
            'note_nlp_source_concept_id': 'ATC',
            'nlp_system': "medext_v3", 
            'nlp_date': f"{datetime.datetime.today():%Y-%m-%d}",
            'nlp_datetime': f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", 
            'term_exists': None,
            'term_temporal': None, 
            'term_modifiers': ','.join(modifiers),
            'validation_level_id': 'automated'
            }     

            res.append(note_nlp_item)

        return res

    