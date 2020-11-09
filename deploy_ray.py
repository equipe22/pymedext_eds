from ray import serve
import requests

from pymedext_eds.extract.utils import load_config
import ray

#from pymedext_eds.med import  Annotator
#from pymedextcore.document import Document
from pymedextcore.annotators import Annotation
import datetime
import pandas as pd
import pkg_resources
from glob import glob

import click
from tqdm import tqdm 

from ray.serve.utils import _get_logger
logger = _get_logger()

from pymedext_eds.med import Pipeline 

from ray.serve.utils import _get_logger
import time
import datetime
import pandas as pd
import math

from pymedext_eds.db import get_engine, get_from_omop_note, get_note_ids, convert_notes_to_doc, load_processed_ids, chunk_to_omop, dump_omop_to_csv
from pymedext_eds.utils import timer, to_chunks
from pymedextcore.document import Document

from pymedext_eds.med import MedicationAnnotator



@ray.remote
def put_request(docs, host = '127.0.0.1', port = '8000', endpoint = 'annotator'):
    
    docs = [x for x in docs if len(x.raw_text()) > 50]
    
    if docs == []:
        return []
    
    #doc += [Document(raw_text= '', ID = '0')]
    
#     json_doc = [
#         {'note_id': d.source_ID,
#         'note_text': d.raw_text()}  for d in doc
#     ]

    json_doc = [doc.to_dict() for doc in docs]
        
    res =  requests.post(f"http://127.0.0.1:8000/annotator", json = json_doc)
    if res.status_code != 200: 
        logger.error(res.status_code)
        logger.error(res.text)
        logger.error([x['note_id'] for x in json_doc])
        #logger.error([x['note_text'] for x in json_doc])
        with open('notes_errors_ray.txt', 'a') as f: 
            for note in json_doc: 
                f.write(f"{note['note_id']}\n")
        #res.raise_for_status()
        return []


    res = res.json()['result']
    
    docs = [Document.from_dict(doc) for doc in res ]

#     for i in range(len(res)):

#         for annot in res[i]['annotations'][1:]:
#             doc[i].annotations.append(Annotation(type=annot["type"],
#                                     value=annot["value"],
#                                     source_ID=annot["source_ID"],
#                                     ID=annot["id"],
#                                     source=annot["source"],
#                                     span=annot["span"],
#                                     attributes=annot["attributes"],
#                                     isEntity=annot["isEntity"]))
        
    return docs    
    
    
@timer
def process_chunk(i, engine, min_date, note_ids, replica_chunk_size = 10, note_nlp_file = None, processed_file =  None):
    
    notes = get_from_omop_note(engine,  min_date = min_date, note_ids = note_ids)
    docs = convert_notes_to_doc(notes)

    try:
        res = ray.get([put_request.remote(c) for c in to_chunks(docs, replica_chunk_size)])
    except Exception as e: 
        print(f"Error: {e}")
        res = None
        
    
    
    if res is not None:
    
        flat_res  = [item for sublist in res for item in sublist]
        note_nlp = chunk_to_omop(flat_res)
    
        logger.info(f'Extracted {note_nlp.shape[0]} rows from chunk {i}')
 
        if note_nlp_file is not None:
            dump_omop_to_csv(note_nlp, note_nlp_file )
            logger.info(f'Appended {note_nlp.shape[0]} rows to {note_nlp_file}')

        if processed_file is not None: 
            with open(processed_file, 'a') as f: 

                for ID in note_ids: 
                    f.write(f"{ID}\n")
    else:
        logger.error('res empty')
        return pd.DataFrame.from_records({})
        
    return note_nlp

@timer
def main_process(limit =1000, chunk_size = 100, min_date = '2020-03-01', replica_chunk_size=10, note_nlp_file = None, processed_file = None):
    
    engine = get_engine()
    
    note_ids = get_note_ids(engine, min_date = min_date)
   
    processed_ids = []
    if processed_file is not None: 
        processed_ids = load_processed_ids(filename = processed_file)
    
    to_process = list(set(note_ids) - set(processed_ids))
    to_process.sort()
    
    if limit is not None:
        to_process = to_process[:limit]
    
    id_chunks = to_chunks(to_process, chunk_size)
    
    n_chunks = len(id_chunks)
    
    logger.info(f'number of chunks to process : {n_chunks}') 
    
    start_time = datetime.datetime.now()
    
    nrows = 0
    for i, ids in enumerate(id_chunks):
        i+=1
        
        tmp = process_chunk(i, engine, min_date, ids, replica_chunk_size, note_nlp_file, processed_file)
        nrows += tmp.shape[0]
        
        time_to_current_chunk = datetime.datetime.now() - start_time
        mean_time = time_to_current_chunk / i
        ETA = mean_time * (n_chunks-i)
        logger.info(f'Processed {i}/{n_chunks} chunks. ETA: {ETA}')
        
    logger.info(f'Process done in {datetime.datetime.now()-start_time}, extracted {nrows} rows')     
        
        
    engine.dispose()
    return nrows  


if __name__ == '__main__':
    
    
    logger = _get_logger()

    num_replicas = 10
    num_gpus = .3
    limit = -1
    chunk_size = 1000
    replica_chunk_size= chunk_size // num_replicas
    note_nlp_file = '../data/omop_tables/test_note_nlp_new.csv'
    processed_file = '../data/omop_tables/notes_processed_med_new.txt'
    min_date='2019-01-01'
    
    client = serve.start()
    
    # ray server
    config = {"num_replicas": num_replicas}
    actor_options = { "num_gpus": num_gpus}
    client.create_backend('annotator', 
                         Pipeline, 
    #                     params,
    #                     postprocess_params,
                         config=config,
                         ray_actor_options=actor_options)
    client.create_endpoint("annotator", backend="annotator", route="/annotator", methods = ['POST'])

    
    # launch client
    main_process(limit =limit, 
                 chunk_size = chunk_size,
                 replica_chunk_size = replica_chunk_size,
                 min_date = min_date,
                 note_nlp_file =note_nlp_file, 
                 processed_file = processed_file)

    
