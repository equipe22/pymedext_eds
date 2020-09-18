#!/export/home/cse180025/.conda/envs/pyenv_cse180025/bin/python

import ray
from ray.serve.utils import _get_logger
import requests
import time
import datetime
import pandas as pd
import math

from pymedext2.db import get_engine, get_from_omop_note, convert_chunk_to_doc
from pymedext2.utils import timer, chunks

from med_server import put_request #, start_server, stop_server
from med_server import  load_processed_ids, chunk_to_omop, dump_omop_to_csv



@timer
def process_chunk(i, raws, big_chunk, note_nlp_file = None, processed = [], processed_file =  None):
    
    docs = convert_chunk_to_doc(raws, big_chunk)
    
    logger.info(f"number of docs in chunk: {len(docs)}")
    
    
    doc_ids = [x.source_ID for x in docs]
    to_process_ids = set(doc_ids) - set(processed)
    chunk = [x for x in docs if x.source_ID in to_process_ids]
    
    logger.info(f"number of docs in chunk after filter of processed ids: {len(chunk)}")

    try:
        res = ray.get([put_request.remote(c) for c in list(chunks(chunk, 10))])
    except Exception as e: 
        print(f"Error: {e}")
    
    if res is not None:
    
        flat_res  = [item for sublist in res for item in sublist]
    
        note_nlp = chunk_to_omop(flat_res)
    
        logger.info(f'Extracted {note_nlp.shape[0]} rows from chunk {i}')
 
        if note_nlp_file is not None:
            dump_omop_to_csv(note_nlp, note_nlp_file )
            logger.info(f'Appended {note_nlp.shape[0]} rows to {note_nlp_file}')

        if processed_file is not None: 
            with open(processed_file, 'a') as f: 

                for ID in to_process_ids: 
                    f.write(f"{ID}\n")
        
    return note_nlp

@timer
def main_process(engine, limit =1000, big_chunk = 100, note_nlp_file = None, processed_file = None):
    
    
    raws = get_from_omop_note(engine, limit, min_date='2020-06-01')
    
    processed_ids = []
    if processed_file is not None: 
        processed_ids = load_processed_ids(filename = processed_file)
    
    #engine.dispose()

    n_chunks = math.ceil(raws.rowcount / big_chunk)
    
    logger.info(f'number of chunks : {n_chunks}') 
    
    start_time = datetime.datetime.now()

    for i,_ in enumerate(range(n_chunks)):
        
        i+=1
        
        nrows = 0
        tmp = process_chunk(i,raws, big_chunk, note_nlp_file = note_nlp_file, processed = processed_ids, processed_file =  processed_file )
        nrows += tmp.shape[0]
    
#         docs = convert_chunk_to_doc(raws, big_chunk)  
#         res = Parallel(n_jobs=num_jobs)(delayed(annotate_docs)(c, pheno_ids) for c in to_chunks(docs, chunksize) )
#         res = flatten(res)
#         note_nlp = chunk_to_omop(res)
#         dump_omop(note_nlp, note_nlp_file )

        time_to_current_chunk = datetime.datetime.now() - start_time
        mean_time = time_to_current_chunk / i
        ETA = mean_time * (n_chunks-i)
        logger.info(f'Processed {i}/{n_chunks} chunks. ETA: {ETA}')
    
    logger.info(f'Process done in {datetime.datetime.now()-start_time}, extracted {nrows} rows')
        
    engine.dispose()
    return nrows



if __name__ == '__main__':
    
    num_replicas = 10
    num_gpus = .32
    limit = -1
    big_chunk = 1000
    note_nlp_file = '../data/omop_tables/note_nlp_med.csv'
    processed_file = '../data/omop_tables/notes_processed_med.txt'

    logger = _get_logger()
    engine = get_engine()
    
    ray.init(address='auto', redis_password='5241590000000000')
    
    #start_server(num_replicas = num_replicas,
    #             num_gpus = num_gpus)
    
    
    main_process(engine, 
                 limit =limit, 
                 big_chunk = big_chunk, 
                 note_nlp_file =note_nlp_file, 
                 processed_file = processed_file)
    
    
    #stop_server()