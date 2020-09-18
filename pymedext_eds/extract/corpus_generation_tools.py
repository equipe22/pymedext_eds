import glob
import pandas as pd
import numpy as np
import seaborn as sns
import os
from os.path import join
import random
import re
import tqdm
import json

def createifnotexists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getfn(path):
    return os.path.splitext(os.path.basename(path))[0]

def get_pymedext_path(text_path):
    return join(os.path.dirname(text_path) , "pymedext", getfn(text_path)+".json")

def get_ent(ann, ent_types):
    return [t for t in ann if t['type'] in ent_types]
    

def get_multiline_format(start, end, mention):
    if re.search("\n", mention):
        scstart, scstop = re.search("\n", mention).span(0)
        span = str(start)+ " " + str(start + scstart) +";"+ str(start + scstop)+ " " + str(end)
        mention = re.sub("\n", "", mention)
    else:
        span = str(start) +" "+ str(end)
        
    return "{}\t{}".format(span, mention)
    

def get_brat_ann(ann, text, pheno_ent_type, threshold=0):
    pheno_ent = get_ent(ann, pheno_ent_type)
    brat = []
    for i, ent in enumerate(pheno_ent):
        start = ent["span"][0]
        end = ent["span"][1]
        mention = text[start:end]

        if ent["attributes"]["score"] < threshold:
            continue
            
        ner_line = "T{}\t{} {}".format(i, ent["type"].split('/')[1], get_multiline_format(start, end, mention))

        if (ent["attributes"]["normalized_mention"]!="") & ("CUI" in ent["attributes"]["normalized_mention"]):
            normalized_mention = ent["attributes"]["normalized_mention"] if "CUI" in ent["attributes"]["normalized_mention"] else ent["attributes"]["normalized_mention"]['0']
            norm_line = "N{}\tReference T{} UMLS_FR:{}\t{}".format(i, i, normalized_mention["CUI"], normalized_mention["STR"])

            if normalized_mention["prob"] < threshold:
                continue
                
            brat.append(norm_line)
            
        brat.append(ner_line)

    return "\n".join(brat)

def get_new_corpus(text_paths, pheno_ent_type, size = 100,  threshold = 0.6, text_min_size = 50, seed=1000000):
    random.seed(seed)
    texts_sample = random.choices(text_paths, k=size)
    corpus = []
    for text_path, pymedext_path in tqdm.tqdm(texts_sample):
        with open(text_path, 'r') as h:
            text = h.read().strip()

        if len(text.split(' ')) < text_min_size:
            continue
        try:
            with open(pymedext_path, "r") as h:
                doc = json.load(h)
        except:
            print(pymedext_path, "not found")
            doc = {'annotations':[]}
            continue

        ann = doc["annotations"]

        brat = get_brat_ann(ann, text, pheno_ent_type, threshold=threshold)
        
        corpus.append((getfn(text_path), text, brat))
        
    return corpus

def write_to_brat(output_dir, corpus):
    for doc_id, text, ann in corpus:
        text_path = join(output_dir, doc_id +".txt")
        ann_path = join(output_dir, doc_id +".ann")

        with open(text_path, 'w') as h:
            h.write(text)

        with open(ann_path, 'w') as h:
            h.write(ann)   


def split_and_write_corpus(corpus, outdir, n_annotator, n_common_files):
    if os.path.exists(outdir):
        os.makedirs(outdir)
        print("Creating dir at {}".format(outdir))
    else:
        print("Warning {} dir already exists".format(outdir))

    #shuffle
    random.shuffle(corpus)
    #take N commone documents
    common_doc = corpus[:n_common_files]
    corpus = corpus[n_common_files:]

    for i, part in enumerate(np.array_split(corpus, n_annotator)):
        subdir = join(outdir, "annotator_{}".format(i))
        createifnotexists(subdir)
        #write annotator_specific
        write_to_brat(subdir, part)

        #write annotator_common
        subdir = join(outdir, "annotator_{}_training".format(i))
        createifnotexists(subdir)
        write_to_brat(subdir, common_doc)


def write_brat_conf(entities, outdir):
    #norm conf
    tool_conf = """[normalization]\nUMLS_FR     DB:data/UMLS_FR, <URL>:https://uts.nlm.nih.gov"""
    
    with open(join(outdir, "tools.conf"), 'w') as h:
        h.write(tool_conf)
        
    #ann conf
    ent, rel, ev, attr = '\n'.join(entities), "", "", ""
    annotation_conf = """[entities]\n{}\n[relations]\n{}\n[events]\n{}\n[attributes]""".format(ent, rel, ev, attr)
    
    with open(join(outdir, "annotation.conf"), 'w') as h:
        h.write(annotation_conf)
        
    #visual conf
    labels = "\n".join([ t+ "|"+ t+ "|" + "".join([u[0] for u in t.split('_')])  for t in entities])
    colors = sns.color_palette(n_colors=len(entities)).as_hex()
    drawing = "[drawing]\n{}".format('\n'.join(["{}\tbgColor:{}".format(et, col) for et, col in zip(entities, colors)]))
    visual_conf = "[labels]\n{}\n\n{}".format(labels, drawing)
    
    
    with open(join(outdir, "visual.conf"), 'w') as h:
        h.write(visual_conf)
        
    #keybord shortcut
    shortcut = []
    for t in entities:
        if t[0] not in shortcut:
            shortcut.append(t[0])
        elif t.split('_')[1][0] not in shortcut:
            shortcut.append(t.split('_')[1][0])
        else:
            shortcut.append("")
            
    shortcut_conf = "\n".join([short+"\t"+t  for t, short in zip(entities, shortcut) if t!=""])

    with open(join(outdir, "kb_shortcuts.conf"), 'w') as h:
        h.write(shortcut_conf)
