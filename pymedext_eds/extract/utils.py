import glob
import re
import pandas as pd
import numpy as np
import os

import yaml


def load_conll(root_path, split):
    data_path = os.path.join(root_path, split+".tsv") 
    header_path = os.path.join(root_path, split+"headers.tsv")
    with open(header_path, "r") as h:
        headers = h.read().strip().split('\t')

    with open(data_path, "r") as h:
        corpus = []
        for sent in h.readlines():
            sent = sent.strip().split('\t')
            if len(sent)==len(headers):
                corpus.append(sent)

    corpus = pd.DataFrame(corpus, columns=headers)
    
    return corpus


def write_conll(dataset, root_path, split):
    data_path = os.path.join(root_path, split+".tsv") 
    header_path = os.path.join(root_path, split+"headers.tsv")
    
    new_dataset = []
    for _, sent in dataset.groupby('sentence_id'):
        new_sent=[]
        for i, row in sent.iterrows():
            row = row.astype('str').str.cat(sep="\t")
            new_sent.append(row)

        new_sent = "\n".join(new_sent)

        new_dataset.append(new_sent+"\n")

    new_dataset = "\n".join(new_dataset)

    with open(data_path, "w") as h:
        h.write(new_dataset)
        
    with open(header_path, "w") as h:
        h.write("\t".join(dataset.columns.tolist()))


def get_paths(root_path):
    # get fn
    text_path = glob.glob(os.path.join(root_path, "*.txt"))
    ann_path = glob.glob(os.path.join(root_path, "*.ann"))

    assert len(text_path) == len(ann_path)

    text_path = [(os.path.relpath(text_path[i], start=root_path)[:-4], path) for i, path in enumerate(text_path)]
    ann_path = [(os.path.relpath(ann_path[i], start=root_path)[:-4], path) for i, path in enumerate(ann_path)]

    text_path = pd.DataFrame(text_path, columns=['rid', 'text_path'])
    ann_path = pd.DataFrame(ann_path, columns=['rid', 'ann_path'])

    file_path = text_path.merge(ann_path, on="rid")
    
    assert len(file_path) == len(text_path)
    
    return file_path

def load_brat(text_path, ann_path):

    with open(text_path, 'r') as f:
        text = f.read()

    with open(ann_path, 'r') as f:
        ann = []
        for line in f.readlines():
            line = line.strip()
            if line[0] !="#":
                ann.append(line.split('\t'))

        ann = pd.DataFrame(ann, columns=['tag', "match", "mention"])
        
    return text, ann

def get_type_begin_end(row):
    
    splitted = row.split()
    et = splitted[0]
    start = int(splitted[1])
    end = int(splitted[-1])

    return et, start, end

def reformat_ann(ann, replace_dict):
    ann = (ann
           .assign(is_entity = lambda x:x.tag.str[0]=="T")
           .assign(is_multiline = lambda x:x.match.str.contains(';'))
          )

    events = ann.loc[lambda x:~x.is_entity]
    entities = ann.loc[lambda x:x.is_entity]

    tmp = pd.DataFrame(entities.match.apply(get_type_begin_end).to_dict(),  index= ['match_type', "start", "end"]).T
    entities = pd.concat([entities, tmp], axis=1, sort=False, ignore_index=False)

    if len(events)>0:
        events_tags = events.match.str.split(':', expand=True).iloc[:,-1].tolist()
        events = entities.loc[lambda x:x.tag.isin(events_tags)]
        entities = entities.loc[lambda x:~x.tag.isin(events_tags)]
    else:
        events = pd.DataFrame([],columns=['tag', 'match', 'mention', 'is_entity', 'is_multiline', 'match_type', 'start', 'end'] )
        
        
    entities = entities.replace({"match_type":replace_dict})
    events = events.replace({"match_type":replace_dict})
    
    if not entities.empty:
        relations = entities.loc[lambda x:x.match_type.str[:3]=="REL"]
        entities = entities.loc[lambda x:x.match_type.str[:3]!="REL"]
    else:
        relations = entities
        
    return relations, entities, events

def get_check_list(entities, text):
    check_list = []
    for ann_i, row_ann in entities.iterrows():
        cand = text[row_ann.start:row_ann.end]
        cand = re.sub("\n", " ", cand).lower()
        check_list.append(cand == row_ann.mention)

    return check_list

def load_config(path):
    #load and reformat params
    with open(path, "r") as h:
        params = yaml.load(h)
    return params
