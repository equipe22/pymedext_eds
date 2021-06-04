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
from ..utils import rawtext_loader

def createifnotexists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getfn(path):
    return os.path.splitext(os.path.basename(path))[0]

def get_pymedext_path(text_path):
    return join(os.path.dirname(text_path) , "pymedext", getfn(text_path)+".json")

def get_ent(doc, ent_types):
    return [ann for ann_type in ent_types for ann in doc.get_annotations(ann_type)]

def brat_norm_to_pdf(dir_path):
    """Make pandas dataframe from BRAT annotations with normalizations info"""
    ann_list = glob.glob(join(dir_path, "*.ann"))
    all_ann = []
    for ann_path in ann_list:
        ann_norm, ann_ner = read_brat_ann(ann_path)
        ann_ner = pd.DataFrame(ann_ner, columns=["doc_id", "ent_id", "ent_type", "start", "stop", "mention"])
        ann_norm =  pd.DataFrame(ann_norm, columns = ["doc_id", "ann_id", "ent_id", "termino", "cui", "mention"])
        ann_ner = ann_ner.merge(ann_norm, how= "left", on = ["doc_id", "ent_id"], suffixes=['_ner', '_norm'])
        all_ann.append(ann_ner)

    all_ann = pd.concat(all_ann)
    
    return all_ann
    
def read_brat_ann(path):
    """Read BRAT ann with normalized annotations"""
    doc_id =  os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    with open(path, "r") as h:
        ann_norm = []
        ann_ner = []
        for line in h.readlines():
            ann_id, field, mention = line.split('\t')
            if ann_id[0] == "T":
                ent_type = field.split(' ')[0]
                start = field.split(' ')[1]
                stop = field.split(' ')[-1]
                ann_ner.append((doc_id, ann_id, ent_type, start, stop, mention.strip()))
            elif ann_id[0] == "N":
                _, ent_id, code = field.split(' ')
                termino, code = code.split(':')
                ann_norm.append((doc_id, ann_id, ent_id, termino, code, mention.strip()))
    
    return(ann_norm, ann_ner)


def get_multiline_format(start, end, mention):
    if re.search("\n", mention):
        scstart, scstop = re.search("\n", mention).span(0)
        span = str(start)+ " " + str(start + scstart) +";"+ str(start + scstop)+ " " + str(end)
        mention = re.sub("\n", "", mention)
    else:
        span = str(start) +" "+ str(end)
        
    return "{}\t{}".format(span, mention)
    

def get_brat_ann(doc, pheno_ent_type, threshold=0, verbose = 0):
    pheno_ent = get_ent(doc, pheno_ent_type)
    text = doc.raw_text()
    brat = []
    for i, ent in enumerate(pheno_ent):
        start = ent.span[0]
        end = ent.span[1]
        mention = text[start:end]
        ent_type = ent.type.split('/')[1]
        if (re.sub('\s','', mention) != re.sub('\s', '', ent.value) ):
            if verbose > 0:
                print(mention, "!!!!", ent.value)
            continue

        if ent.attributes["score"] < threshold:
            continue
            
        #brat line
        ner_line = "T{}\t{} {}".format(i, ent_type, get_multiline_format(start, end, mention))

        #norm form
        norm_type = "umls_" + ent_type.lower()
        norm_forms = [anno for anno in doc.annotations if (anno.type == norm_type) & (anno.source_ID==ent.ID) ]
        
        if norm_forms:
            norm_form = norm_forms[0]
            normalized_mention = norm_form.attributes['label']
            cui = norm_form.attributes['cui']
            norm_score =  norm_form.attributes['score']

            norm_line = "N{}\tReference T{} UMLS_FR:{}\t{}".format(i, i, cui, normalized_mention)
            if norm_score < threshold:
                continue

            brat.append(norm_line)
            
        brat.append(ner_line)

    return "\n".join(brat), text



def get_new_corpus(text_path, pipeline, pheno_ent_type, size = None,  threshold = 0,  min_n_tokens = 0, seed=1000000, verbose = 0):
    """Generate a pre-annotated corpus for entity and normalizaton from directory of text file, using a pymedext pipeline, and filtering by entity types"""
    file_list = glob.glob(text_path + '/*.txt')
    random.seed(seed)
    if size is None:
        size = len(file_list)
    file_list = random.choices(file_list, k=size)
    
    docs = [rawtext_loader(x) for x in file_list]
    
    for doc in docs:
        doc.annotate(pipeline)
        
    corpus = []
    for doc in docs:
        brat_ann, text = get_brat_ann(doc= doc, pheno_ent_type = ["ENT/SIGNS", "ENT/DIAG_NAME"], threshold=threshold, verbose = verbose)
        if len(text.split(' ')) < min_n_tokens:
            continue
            
        corpus.append((doc.source_ID, text, brat_ann))
        
    return corpus

def write_to_brat(output_dir, corpus):
    createifnotexists(output_dir)

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



def write_brat_conf(outdir = None, entities = [], relations = {}, norm = {}):
    """
    Generate and write configs for BRAT annotations tools
    #entities
    list: ["ENT/SIGNS", "ENT/DIAG_NAME"]
    #relations
    Located	Arg1:Person,	Arg2:Building|City|Country
    #norm
    dict of dic: "UMLS_ALL":{"DB":"umls0/umls_bglinsty", "<URL>":"http://en.wikipedia.org", "<URLBASE>":"http://en.wikipedia.org/?curid=%s"}
    """
    #norm conf
    if norm:
        tool_conf = "[normalization]\n" + "\n".join([t + "\t" + ", ".join([k + ":" + v for k,v in norm[t].items()] )for t in norm.keys()])
        if outdir:
            with open(join(outdir, "tools.conf"), 'w') as h:
                h.write(tool_conf)
                
        else:
            print(tool_conf)
        
    #ann conf
    ent, rel, ev, attr = '\n'.join(entities), "", "", ""
    rel = ""
    for k, v in relations.items():
        rel += "{}\tArg1:{},\tArg2:{}\n".format(k, "|".join(v['Arg1']), "|".join(v['Arg2']))
    
    annotation_conf = """[entities]\n{}\n[relations]\n{}\n[events]\n{}\n[attributes]""".format(ent, rel, ev, attr)
    if outdir:
        with open(join(outdir, "annotation.conf"), 'w') as h:
            h.write(annotation_conf)
    else:
        print(annotation_conf)
        
        
    #visual conf
    labels = "\n".join([ t+ "|"+ t+ "|" + "".join([u[0] for u in t.split('_')])  for t in entities])
    colors = sns.color_palette(n_colors=len(entities)).as_hex()
    drawing = "[drawing]\n{}".format('\n'.join(["{}\tbgColor:{}".format(et, col) for et, col in zip(entities, colors)]))
    visual_conf = "[labels]\n{}\n\n{}".format(labels, drawing)
    
    if outdir:
        with open(join(outdir, "visual.conf"), 'w') as h:
            h.write(visual_conf)
            
    else:
        print(visual_conf)
        
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

    if outdir:
        with open(join(outdir, "kb_shortcuts.conf"), 'w') as h:
            h.write(shortcut_conf)
    else:
        print(shortcut_conf)
