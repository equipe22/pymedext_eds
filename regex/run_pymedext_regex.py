#!/export/home/cse180025/.conda/envs/pyenv_cse180025/bin/python
from pprint import pprint
import glob
import datetime

from pymedext_eds.annotators import Endlines, SentenceTokenizer, Hypothesis, ATCDFamille, SyntagmeTokenizer, Negation, RegexMatcher, rawtext_loader
from pymedext_core.document import Document

from pymedext_eds.db import get_engine, construct_query

import math
import pandas as pd
import functools
import time

from joblib import Parallel, delayed
import multiprocessing

from logzero import logger, logfile#, setup_default_logger

from pymedext_eds.utils import timer

# def timer(func):
#     """Print the runtime of the decorated function"""
#     @functools.wraps(func)
#     def wrapper_timer(*args, **kwargs):
#         start_time = time.perf_counter()    # 1
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()      # 2
#         run_time = end_time - start_time    # 3
#         logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
#         return value
#     return wrapper_timer


# def construct_query(limit, min_date='2020-03-01', view='unstable'): 
#     if limit != -1:
#         limit = "LIMIT {}".format(limit)
#     else:
#         limit = ""
        
#     query = f"""
#         SELECT note_id, person_id, note_text, note_date
#         FROM {view}.note 
#         WHERE note_date > '{min_date}'
#         AND note_text IS NOT NULL OR note_text not in ('','\n', ' ') 
#         {limit}
#         """
#     #logger.debug('query set with parameters: view = {}, min_date= {}, limit={}'.format(view, min_date, limit))
#     return query

@timer
def get_from_omop_note(engine, limit = -1, min_date='2020-03-01', note_ids= None):

    query = construct_query(limit, min_date, note_ids = note_ids)
    return engine.execute(query)
    #return pd.read_sql_query(query, engine)


def convert_chunk_to_doc(query_res, chunksize): 
    res = []
    
    for note_id, person_id, raw_text, note_date in query_res.fetchmany(chunksize): 
        res.append(Document(raw_text=raw_text,
                            ID = note_id, 
                            attributes = {'person_id':person_id}, 
                            documentDate = note_date.strftime("%Y/%m/%d")
        ))
    return res


@timer
def get_note_nlp_pheno_ids(engine):
    pheno_ids = pd.read_sql_query("""
        select distinct note_id
        from unstable.note_nlp_pheno
    """, engine)
    
    pheno_ids = pheno_ids.note_id.to_list()
    
    return set([int(x) for x in pheno_ids if not math.isnan(x)] )

    
def to_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def flatten(l):
    """Flatten List of Lists to List"""
    return [item for sublist in l for item in sublist]


def init_note_nlp(note_nlp_file): 
    
    new_cols = ['note_nlp_id',
    'note_id',
    'person_id',
    'section_concept_id',
    'snippet',
    'offset_begin',
    'offset_end',
    'lexical_variant',
    'note_nlp_concept_id',
    'note_nlp_source_concept_id',
    'nlp_system',
    'nlp_date',
    'nlp_datetime',
    'term_exists',
    'term_temporal',
    'term_modifiers',
    'validation_level_id',
    'context',
    'negation',
    'certainty']
    
    note_nlp = pd.DataFrame(columns = new_cols)
    note_nlp.to_csv(note_nlp_file,index = False, mode = 'w')

def doc_to_omop(annotated_doc, annotation_type = 'regex'): 
    
    annots = annotated_doc.get_annotations(annotation_type)
    if annots == []:
        return []
    
    note_id = annotated_doc.source_ID
    person_id = annotated_doc.attributes.get('person_id')

    regex = [x.to_dict() for x in annots]

    res = []

    for r in regex: 
        res.append({
        'note_nlp_id':None ,
        'note_id':note_id,
        'person_id':person_id, 
        'section_concept_id':None ,
        'snippet':r['attributes'].get('snippet',None),
        'offset_begin':r['span'][0],
        'offset_end':r['span'][1],
        'lexical_variant':r['value'] ,
        'note_nlp_concept_id':r['attributes'].get('label') , 
        'note_nlp_source_concept_id': r['attributes'].get('id_regexp'),
        'nlp_system':r['attributes'].get('version') ,
        'nlp_date': f"{datetime.datetime.today():%Y-%m-%d}",
        'nlp_datetime':f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}" ,
        'term_exists' : None,
        'term_temporal': None,
        'term_modifiers' : f'"context"="{r["attributes"].get("context")}";"certainty"="{r["attributes"].get("hypothesis")}";"negation"="{r["attributes"].get("negation")}"' ,
        'validation_level_id':None , 
        'context':r['attributes'].get('context'),
        'negation':r['attributes'].get('negation'),
        'certainty':r['attributes'].get('hypothesis')
        })

    return res

@timer
def chunk_to_omop(annotated_chunk, annotation_type='regex'): 
    note_nlp = []
    for doc in annotated_chunk:
        note_nlp += doc_to_omop(doc)
    return pd.DataFrame.from_records(note_nlp)

def dump_omop_to_csv(note_nlp, filename, mode = 'a'): 
    note_nlp.to_csv(filename,index = False, header = False, mode = mode)
    
@timer
def annotate_docs(docs, pheno_ids,annotators):
    to_process_ids = set([x.source_ID for x in docs]) - pheno_ids
    docs_filtered = [x for x in docs if x.source_ID in to_process_ids]
    for doc in docs_filtered:
        doc.annotate(annotators)
    return docs_filtered

# @timer
# def process_docs(raws, chunksize, pheno_ids): 
#     docs = convert_chunk_to_doc(raws, chunksize)
#     return annotate_docs(docs, pheno_ids)

@timer
def process_chunk(i,raws, big_chunk, num_jobs, pheno_ids, note_nlp_file, chunksize, annotators): 
    
    docs = convert_chunk_to_doc(raws, big_chunk)
    
    res = Parallel(n_jobs=num_jobs)(delayed(annotate_docs)(c, pheno_ids, annotators) for c in to_chunks(docs, chunksize) )
    res = flatten(res)
    note_nlp = chunk_to_omop(res)
    logger.info(f'Extracted {note_nlp.shape[0]} rows from chunk {i}')
 
    if note_nlp_file is not None:
        dump_omop_to_csv(note_nlp, note_nlp_file )
        logger.info(f'Appended {note_nlp.shape[0]} rows to {note_nlp_file}')

    return note_nlp


@timer
def main_process(engine,annotators, limit =1000, big_chunk = 100, chunksize = 10, num_jobs = 10, note_nlp_file = None):
    raws = get_from_omop_note(engine, limit)
    #pheno_ids = get_note_nlp_pheno_ids(engine)
    pheno_ids = set([])
    engine.dispose()

    n_chunks = math.ceil(raws.rowcount / big_chunk)
    #raws = raws.to_dict(orient='records')
    #big_chunks = to_chunks(raws, big_chunk)
    
    logger.info(f'number of chunks : {n_chunks}') 
    
    start_time = datetime.datetime.now()

    for i,_ in enumerate(range(n_chunks)):
        
        i+=1
        
        nrows = 0
        tmp = process_chunk(i,raws, big_chunk, num_jobs,pheno_ids, note_nlp_file, chunksize, annotators )
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
    
    log_file = '/export/home/cse180025/prod_information_extraction/logs/run_pymedext_regex.log'
    logfile(log_file, maxBytes = 1e6, backupCount=3, mode ='a')
    
    
    endlines = Endlines(['raw_text'], 'endlines', 'endlines:v1')
    sentences = SentenceTokenizer(['endlines'], 'sentence', 'sentenceTokenizer:v1')
    hypothesis = Hypothesis(['sentence'], 'hypothesis', 'hypothesis:v1')
    family = ATCDFamille(['sentence'], 'context', 'ATCDfamily:v1')
    syntagmes = SyntagmeTokenizer(['sentence'], 'syntagme', 'SyntagmeTokenizer:v1')
    negation = Negation(['syntagme'], 'negation', 'Negation:v1')
    regex = RegexMatcher(['endlines','syntagme'], 'regex', 'RegexMatcher:v1', 'list_regexp.json')
    annotators=[endlines, sentences, hypothesis, family, syntagmes, negation, regex]
    
    engine = get_engine()
    
    limit =-1
    big_chunk = 20000
    chunksize = 1000
    num_jobs = 20
    note_nlp_file = '/export/home/cse180025/prod_information_extraction/data/omop_tables/note_nlp_pheno.csv'
    init = True
    
    if (init):
        init_note_nlp(note_nlp_file)
    
    main_process(engine = engine,
                 annotators = annotators,
                 limit =limit, 
                 big_chunk = big_chunk, 
                 chunksize = chunksize, 
                 num_jobs = num_jobs, 
                 note_nlp_file = note_nlp_file)
    
    
