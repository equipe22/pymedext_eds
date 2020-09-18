#from joblib import Parallel, delayed
#import glob
import os
from os.path import join
import pandas as pd
#from tqdm import tqdm
from .normalisation import clean_freq, clean_dose, clean_drug, clean_class
from ..pyromedi.romedi import Romedi
from pymedext_core.document import Document
from pymedext_core.annotators import Annotation
#from pymedext.src.pymedext.document import Document, Annotation 
import json
from .atcd_classifier import atcd_classifier, atcd_features_creation
from copy import deepcopy
import git

def get_fn(path):
    return os.path.splitext(os.path.basename(path))[0]


class ConllPostProcess(object):
    def __init__(self,
                 ent_tags,
                 rel_tags=[],
                 other_tags = [],
                 anchor_class = ["ENT/DRUG", "ENT/CLASS"],
                 text_suffix = "",
                 normalize = False,
                 verbose = 0,
                 romedi_path = "pymedext/cache_Romedi2-2-0.p",
                 class_norm = "data/terminologies/drug_classes/normalisation_classes.json",
                 atcd_model_path = "",
                 input_dir = "",
                 get_neg_score = False):
        
        self.ent_tags = ent_tags
        self.rel_tags = rel_tags
        self.tag_names = ent_tags + rel_tags + other_tags
        self.anchor_class = anchor_class
        self.other_tags = other_tags
        self.normalize = normalize
        self.get_neg_score = get_neg_score
        
        #model for predicting if diagnosis is medical history
        if atcd_model_path !="":
            text = "sent_text"
            features = ["section_type", "relative_loc"]
            features.append(text)
    
            self.default_features = features
            self.atcd_classifier = atcd_classifier(text, self.default_features).load_model(atcd_model_path)
            self.atcd_featurizer = atcd_features_creation(join(input_dir, "document_references.csv"),
                                                          input_dir, 
                                                          self.default_features)

            self.infer_atcd = True
        else:
            self.infer_atcd = False
            
            
        #do normalization of entities
        if normalize:
            self.romedi = Romedi(from_cache=romedi_path)
            with open(class_norm, 'r') as f:
                self.class_norm = json.load(f)
                
            self.cache_mentions = {"ENT/DRUG":{}, 
                                   "ENT/CLASS":{},
                                   "ENT/FREQ":{},
                                   "ENT/DOSE":{}}
            
        #to get raw mention of entites (without tokenization)
        if text_suffix != "":
            watch = pd.read_csv(join(input_dir, "watch.csv"))
            self.text_paths = watch.set_index('id').loc[:,"path"].to_dict()
        else:
            self.text_paths = {}
            
        #for versioning
        # repo = git.Repo(search_parent_directories=True)
        # self.sha = repo.head.object.hexsha
        self.sha = 'medext_v3'
        self.verbose = verbose

        #default drug entity fields
        self.normed_entity = {"anchor":{}, 
                             'ENT/DOSE': [],
                             'ENT/FREQ': [],
                             'ENT/DURATION': [],
                             'ENT/CONDITION': [],
                             'ENT/ROUTE': [],
                             "EV/CONTINUE": [],
                             "EV/DECREASE": [],
                             "EV/INCREASE":[],
                             "EV/START":[],
                             "EV/START_STOP":[],
                             "EV/STOP":[],
                             "EV/SWITCH":[]
                                    }
            

    def log(self, msg):
        if self.verbose>0:
            print(msg)
            
    def to_pymedext(self, doc_text, brat_doc):
        doc_id = brat_doc.pop("doc_id")
        doc = Document(raw_text = "", ID = doc_id)

        #for each annotations
        for ann_id, ann_value in brat_doc.items():
            #TODO: sections span are not well computed as computed only on saved sentences in the conll csv
            #get sections
            if str(ann_id)[:3] == 'sec': 
                pymedext_ann = Annotation(type= 'section',
                                         value= ann_value['section_type'],
                                         source = self.sha, 
                                         source_ID = doc_id, 
                                         span = ann_value['span'])
                doc.annotations.append(pymedext_ann)
                

            #if sentences
            elif str(ann_id)[:4] == 'sent': 
                #sget entences
                sent_annot = Annotation(type = 'sentence',
                                        value = ann_value.pop("mention"), 
                                        source = self.sha, 
                                        source_ID = doc_id,
                                        span = ann_value.pop('span'),
                                        attributes = {'raw_mention':ann_value.pop('raw_mention')}
                                        

                                       )
                doc.annotations.append(sent_annot)

                #for each entity in sentence
                for ent_id, entity in ann_value.items():
                    #entity type pheno
                    if ent_id[:5] == "pheno":
                        ent= Annotation(type = entity['ent_type'], 
                                        value = entity['tokens'],
                                        source = self.sha, 
                                        source_ID = sent_annot.ID,
                                        span = entity['span'],
                                        attributes = {'section_type': [{'raw_mention': entity['section_type']}],
                                                      'normalized_mention':entity['normalized_mention'],
                                                      'score':entity["score"],
                                                      'neg_score':entity["neg_score"]
                                                     })
                        doc.annotations.append(ent)

                    #entity type drugblob
                    elif ent_id[:8] == "drugblob":
                        if not entity['anchor']:
                            continue
                        ent= Annotation(type = entity['anchor']['ent_type'], 
                                        value = entity['anchor']['tokens'],
                                        source = self.sha, 
                                        source_ID = sent_annot.ID,
                                        span = entity['anchor']['span'],
                                        attributes = {'section_type': [{'raw_mention': entity['anchor']['section_type']}],
                                                      'normalized_mention':entity["anchor"]['normalized_mention'],
                                                      'score':entity['anchor']["score"],
                                                      'neg_score':entity['anchor']["neg_score"],
                                                      'ENT/CONDITION':entity['ENT/CONDITION'],
                                                      'ENT/DOSE':entity['ENT/DOSE'],
                                                      'ENT/DURATION':entity['ENT/DURATION'],
                                                      'ENT/FREQ':entity['ENT/FREQ'],
                                                      'ENT/ROUTE':entity['ENT/ROUTE']})

                        doc.annotations.append(ent)
                    else:
                        raise NotImplementedError
                                        
        return doc
    
    
    def get_group(self, conll):
        conll['tok'] = conll['tok'].fillna("")
        for leave in self.tag_names:
            conll.loc[:, leave+"__id"] = -1
            conll.loc[lambda x:x[leave] != "O", leave+"__id"] = (conll.loc[lambda x:x[leave] != "O", leave].str[0] == "B").cumsum()
            
        return conll
    
    
    def clean(self, mention, ent_type, context= ""):
        if ent_type == "ENT/FREQ":
            if mention.lower() in self.cache_mentions[ent_type].keys(): 
                return self.cache_mentions[ent_type][mention.lower()]
            
            else: 
                cleaned =  clean_freq(mention)
                self.cache_mentions[ent_type][mention.lower()] = cleaned
                return cleaned
        
        elif ent_type == "ENT/DOSE":
            if mention.lower() in self.cache_mentions[ent_type].keys(): 
                return self.cache_mentions[ent_type][mention.lower()]
            else: 
                cleaned =  clean_dose(mention)
                self.cache_mentions[ent_type][mention.lower()] = cleaned
                return cleaned
        
        elif ent_type == "ENT/DRUG":
            if mention.lower() in self.cache_mentions[ent_type].keys(): 
                return self.cache_mentions[ent_type][mention.lower()]
            else: 
                cleaned =  clean_drug(mention, self.romedi)
                self.cache_mentions[ent_type][mention.lower()] = cleaned
                return cleaned
        elif ent_type == 'ENT/CLASS':
            cleaned = clean_class(mention, self.class_norm)
            return cleaned
            
        else:
            return ""
        
    def pp_pheno(self, brat, sent_id, sentence, raw_text):
        if sent_id not in brat:
            span = (int(sentence["start"].min()), int(sentence["end"].max()))
            sent_entity =  {"mention":' '.join(sentence['tok'].tolist()),
                            "raw_mention":raw_text[span[0]:span[1]] if raw_text else None,
                            "span":span}
            
            brat.update({sent_id:sent_entity})
        for ent_id, ent in sentence.groupby("pheno_pred__id"):
            if ent_id != -1:
                #define entity
                entity = self.conll2dict("pheno_pred", ent, raw_text)
                brat[sent_id].update({"pheno_{}".format(ent_id):entity})
            
        return brat

    def conll2dict(self, col_name, conll, raw_text):
        tokens = conll['tok'].tolist()
        scores = conll[col_name+"_score"].tolist()
        scores = [float(score) for score in scores]
        score = sum(scores)/len(scores)
        mention = " ".join(tokens)
        span = (int(conll["start"].min()), int(conll["end"].max()))
        ent_type = conll[col_name].str[2:].values.item(0)
        section_type = conll['section_tag'].str[2:].values.item(0)
        if self.get_neg_score:
            neg_scores = [float(score) if pred !="O" else 1-float(score)  for score, pred in zip(conll["neg_pred_score"], conll["neg_pred"])]
            neg_score = sum(neg_scores)/len(neg_scores)
        else:
            neg_scores = []
            neg_score = 0
            
        entity = {"tokens":tokens,
                  "scores":scores,
                  "score":score,
                  "neg_scores":neg_scores,
                  "neg_score":neg_score,
                  "mention": mention, 
                  "raw_mention":raw_text[span[0]:span[1]] if raw_text else None,
                  "normalized_mention":self.clean(mention, ent_type) if self.normalize else "",
                  "span":span,
                  "ent_type":ent_type,
                  "section_type":section_type}
        
        return entity
        
        
    def pp_drug_blob(self, brat, sent_id, sentence, raw_text):
        span = (int(sentence["start"].min()), int(sentence["end"].max()))
        sent_entity =  {"mention":' '.join(sentence['tok'].tolist()),
                        "raw_mention":raw_text[span[0]:span[1]] if raw_text else None,
                        "span":span}

        brat.update({sent_id:sent_entity})
        for drug_id, drugblob in sentence.groupby("drugblob_pred__id"):
            drug_id = "drugblob_{}".format(drug_id)
            #accumulate entities and envents
            brat[sent_id].update({drug_id:{}})
            #entities
            for _, (ent_id, ent) in enumerate(drugblob.groupby("entity_pred__id")):
                #inside entities
                if ent_id !=-1:
                    #define entity
                    entity = self.conll2dict("entity_pred", ent, raw_text)
                    brat[sent_id][drug_id].update({"ent_{}".format(ent_id):entity})

                else:
                    #outside entities
                    pass

            #events
            for _, (ent_id, ent) in enumerate(drugblob.groupby("event_pred__id")):
                #inside events
                if ent_id != -1:
                    entity = self.conll2dict("event_pred", ent, raw_text)
                    brat[sent_id][drug_id].update({"ev_{}".format(ent_id):entity})
                else:
                    #outside events
                    pass

            #aggregate entities and events in a drug blob
            brat_drugblob = brat[sent_id].pop(drug_id)
            if not brat_drugblob:
                pass
            #if inside a drugblob
            elif (drug_id != -1):
                #default
                normed_entity = deepcopy(self.normed_entity)
                #reverse sort
                brat_drugblob = {db_k:brat_drugblob[db_k] for db_k in reversed(sorted(brat_drugblob.keys()))}
                #update
                for bdb_k, ent_i in brat_drugblob.items():
                    if ent_i["ent_type"] in self.anchor_class:
                        normed_entity["anchor"] = ent_i
                        normed_k = bdb_k
                    else:
                        normed_entity[ent_i.pop("ent_type")] += [ent_i]
                    
                brat[sent_id][drug_id] = normed_entity

            #drug outside drug blob : keep only drug not isolated freq or other
            else:
                new_bratdrugblob = {}
                for bdb_k, ent_i in brat_drugblob.items():
                    normed_entity = deepcopy(self.normed_entity)
                    if ent_i["ent_type"] in self.anchor_class:
                        normed_entity["anchor"] = ent_i
                        new_bratdrugblob.update({bdb_k:normed_entity})
                        brat[sent_id]["{}_{}".format(drug_id, bdb_k)] = new_bratdrugblob

        return brat

    
    def pp_doc(self, doc_id, doc_conll):
        #document level
        brat = {"doc_id":doc_id}
        if doc_id in self.text_paths:
            with open(self.text_paths[doc_id], 'r') as h:
                raw_text = h.read().strip()
        else:
            raw_text = ""
                
        #upper to sentence level
        if self.other_tags:
            for sec_id, sec in doc_conll.groupby(self.other_tags[0]+'__id'):
                span = (int(sec["start"].min()), int(sec["end"].max()))
                section_type = sec['section_tag'].str[2:].values.item(0)
                brat_sec = {
                    "span":span,
                    "section_type":section_type
                }
                
                brat.update({"sec_{}".format(sec_id):brat_sec})

                
        #sentence level
        for sent_id, sentence in doc_conll.groupby("sentence_id"):
            sent_id = "sent_{}".format(sent_id)
            brat = self.pp_drug_blob(brat, sent_id, sentence, raw_text)
            if "pheno_pred" in self.ent_tags:
                brat = self.pp_pheno(brat, sent_id, sentence, raw_text)
            
            
        pymedext = self.to_pymedext(raw_text, brat).to_dict()
        if self.infer_atcd:
            pymedext = self.get_atcd_attr(pymedext)
                        
        return pymedext

        

    def get_atcd_attr(self, pymedext):
        data = pd.DataFrame(pymedext["annotations"]) # On lit le JSON
        sentences = pd.DataFrame.copy(data)
        data = data[data["type"] == "ENT/DIAG_NAME"] # On sélectrionne les types ENT/DIAG_PROC

        if data.empty:
            return pymedext

        sentences = sentences[sentences["type"] == "sentence"] # On sélectrionne les types ENT/DIAG_PROC

        #reformat
        data["section_type"] = data["attributes"].apply(lambda x: x["section_type"][0]["raw_mention"]) # section_type
        data = (data
                .join(sentences[["value","span","id", "source_ID"]]
                      .set_index("id"), on = "source_ID", rsuffix = "_sentence")
                .rename({"value_sentence": "sent_text", "source_ID_sentence": "doc_id"}, axis = 'columns') # doc_ID et sent_text
               )

        data["doc_id"] = data["doc_id"].str.split("_", expand = True)[1].astype(int) #doc_id


        
        

        #featurize:
        data, features = self.atcd_featurizer.transform(data)
        self.atcd_featurizer.features = self.default_features
        #infer
        self.atcd_classifier.features = features
        data["infer_atcd"] = self.atcd_classifier.predict_proba(data)[:, 1]
        
        #append to pymedext
        for ann in pymedext['annotations']:
            if ann['type']=="ENT/DIAG_NAME":
                row = data.loc[data.id==ann["id"]]
                ann["attributes"]['atcd_score'] = row.infer_atcd.values.item()
        
        return pymedext
        
        
    def pp_conll(self, conll, id_var = "encounter_num"):
        conll = self.get_group(conll)
        sentences_tags = []
        for k, doc in conll.groupby(id_var):
            sentences_tags.append(self.pp_doc(k, doc))

        return sentences_tags
    
            
    def pp_conll_simple(self, conll):
        conll = self.get_group(conll)
        sentences_tags = []
        for k, doc in conll.groupby("note_id"):
            sentences_tags.append(self.pp_doc(k, doc))

        return sentences_tags
    
    

