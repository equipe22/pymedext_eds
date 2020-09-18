#!/export/home/cse180025/.conda/envs/pyenv_cse180025/bin/python

from pymedext_eds.extract.utils import load_config
import ray
from ray import serve
import requests
from pymedext_eds.med import  Annotator
from pymedextcore.document import Document
from pymedextcore.annotators import Annotation
from pymedext_eds.utils import timer
import datetime
import pandas as pd

import click

from ray.serve.utils import _get_logger
logger = _get_logger()

host = '127.0.0.1'
port = 8000

@ray.remote
def put_request(doc, host = '127.0.0.1', port = '8000', endpoint = 'annotator'):
    
    doc = [x for x in doc if len(x.raw_text()) > 50]
    
    if doc == []:
        return []
    
    #doc += [Document(raw_text= '', ID = '0')]
    
    json_doc = [
        {'note_id': d.source_ID,
        'note_text': d.raw_text()}  for d in doc
    ]
        
    res =  requests.post(f"http://{host}:{port}/{endpoint}", json = json_doc)
    if res.status_code != 200: 
        logger.error(res.status_code)
        logger.error(res.text)
        logger.error([x['note_id'] for x in json_doc])
        #logger.error([x['note_text'] for x in json_doc])
        with open('notes_errors.txt', 'a') as f: 
            
            for note in json_doc: 
                f.write(f"{note['note_id']}\n")
        #res.raise_for_status()
        return None


    res = res.json()['result']

    for i in range(len(res)):

        for annot in res[i]['annotations'][1:]:
            doc[i].annotations.append(Annotation(type=annot["type"],
                                    value=annot["value"],
                                    source_ID=annot["source_ID"],
                                    ID=annot["id"],
                                    source=annot["source"],
                                    span=annot["span"],
                                    attributes=annot["attributes"],
                                    isEntity=annot["isEntity"]))
        
    return doc

def init():
     ray.init(address='auto', redis_password='5241590000000000')

@click.command()
@click.argument("arg", type=click.STRING)
@click.option('--config_file', default="/export/home/cse180025/prod_information_extraction/pymedext_eds/configs/deploy.yaml", help='path to the config file')
def main(arg, config_file): 

    if arg == 'start':
        start(config_file)
    elif arg == 'stop':
        stop()
    
def stop(endpoint = 'annotator'):
    #serve.init(http_host= host, http_port = port)
    init()
    serve.init()
    
    serve.delete_endpoint(endpoint)
    serve.delete_backend(endpoint)

def start(config_file
#     infer_config = "/export/home/cse180025/prod_information_extraction/configs/infer_configs/ray_v2.yaml",
#     post_config = '/export/home/cse180025/prod_information_extraction/configs/postp_configs/covid_v2_ray.yaml',              
#     num_replicas = 10,
#     num_gpus = .32
         ):
    
    cfg = load_config(config_file)

    params = load_config(cfg['infer_config'])
    postprocess_params = load_config(cfg['post_config'])
    
    init()
    
    serve.init()
    
    config = {"num_replicas": cfg['num_replicas']}
    actor_options = { "num_gpus": cfg['num_gpus']}
    serve.create_backend('annotator', 
                         Annotator, 
                         params,
                         postprocess_params,
                         config=config,
                         ray_actor_options=actor_options)
    serve.create_endpoint("annotator", backend="annotator", route="/annotator", methods = ['POST'])
    
    

def chunk_to_omop(annotated_chunk): 
    
    omop = [doc_to_omop(x) for x in annotated_chunk]
    omop = [item for sublist in omop for item in sublist]
    
    return pd.DataFrame.from_records(omop)

def dump_omop_to_csv(note_nlp, filename, mode = 'a'): 
    note_nlp.to_csv(filename,index = False, header = False, mode = mode)
    
    
def load_processed_ids(filename = 'data/omop_tables/notes_processed_med.txt'): 
    processed = []
    with open(filename, 'r') as f:
        for ID in f: 
            processed.append(int(ID.strip()))
            
    return list(set(processed))


def doc_to_omop(annotated_doc): 
    
    annots = [x.to_dict() for x in annotated_doc.get_annotations('ENT/DRUG') + annotated_doc.get_annotations('ENT/CLASS') ]
    sentences = [x.to_dict() for x in annotated_doc.get_annotations('sentence')]
    if annots == []:
        return []
    
    note_id = annotated_doc.source_ID
    person_id = annotated_doc.attributes.get('person_id')
    
    

    #regex = [x.to_dict() for x in annots]

    res = []

    for drug in annots: 
        
        section = drug["attributes"]['section_type'][0]['raw_mention']
        sentence = [x['value'] for x in sentences if x['id'] == drug['source_ID']][0]
        
        norm = None
        if 'normalized_mention' in drug["attributes"].keys(): 
            if (drug['attributes']['normalized_mention'] != []) & (drug['attributes']['normalized_mention'] != {}):
                if drug['type'] == 'ENT/DRUG':
                    norm = drug['attributes']['normalized_mention']['ATC7']
                else: 
                    norm = drug['attributes']['normalized_mention']

        dose = None
        dose_norm = None
        route = None
        duration = None 
        duration_norm = None
        freq = None
        freq_norm = None

        modifiers = []
        if drug['attributes'] != []: 
            for k,v in drug['attributes'].items():

                if k not in ['normalized_mention', 'score', 'neg_score']:
   #                 logger.info(v)
                    for vv in v:
                        if 'mention' in vv.keys():
                            modifiers.append("{}='{}'".format(k, vv['mention']))
                            if k == 'ENT/ROUTE': 
                                route= vv['mention']
                            if k == 'ENT/DOSE':
                                dose == vv['mention']
                            if k== 'ENT/DURATION':
                                duration = vv['mention']
                            if k== 'ENT/FREQ':
                                freq = vv['mention']

                            if 'normalized_mention' in vv.keys():
                                modifiers.append("{}_norm='{}'".format(k, vv['normalized_mention']))
                                if k == 'ENT/DOSE': 
                                    dose_norm = vv['normalized_mention']                               
                                if k == 'ENT/DURATION':
                                    duration_norm= vv['normalized_mention']
                                if k == 'ENT/FREQ':
                                    freq_norm= vv['normalized_mention']
                else:
                    if k == 'score':
                        modifiers.append("score={}".format(v))
                    if k == 'neg_score':
                        modifiers.append("neg_score={}".format(v))
                        
        note_nlp_item = {
        'note_nlp_id' : None,
        'note_id': note_id,
        'person_id': person_id,
        'section_concept_id': section,
        'snippet': sentence, 
        'offset_begin': drug['span'][0],
        'offset_end': drug['span'][1], 
        'lexical_variant': ' '.join(drug['value']),
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


if __name__ == '__main__':
    
    main()
    