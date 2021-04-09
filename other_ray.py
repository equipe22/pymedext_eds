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

from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer
import pkg_resources
from glob import glob


class Pipeline:
    """
    Pipeline designed to work with Ray 1.2.0
    """
    
    def __init__(self, device=None):

        if device is None:
            if ray.get_gpu_ids():
                device = f'cuda'
            else:
                device = 'cpu'

        self.endlines = Endlines(["raw_text"], "clean_text", ID="endlines")
        self.sections = SectionSplitter(['clean_text'], "section", ID='sections')
        self.sentenceSplitter = SentenceTokenizer(["section"], "sentence", ID="sentences")

        self.models_param = [
            {
                'tagger_path': 'data/models/apmed5/entities/final-model.pt',
                'tag_name': 'entity_pred'
            },
            {
                'tagger_path': 'data/models/apmed5/events/final-model.pt',
                'tag_name': 'event_pred'
            },
            {
                'tagger_path': "data/models/apmed5/drugblob/final-model.pt",
                'tag_name': 'drugblob_pred'
            },
        ]

        self.med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=self.models_param, device=device)

        data_path = pkg_resources.resource_filename('pymedext_eds', 'data/romedi')
        romedi_path = glob(data_path + '/*.p')[0]

        self.norm = MedicationNormalizer(['ENT/DRUG', 'ENT/CLASS'], 'normalized_mention', ID='norm',
                                         romedi_path=romedi_path)

        self.pipeline = [self.endlines, self.sections, self.sentenceSplitter, self.med, self.norm]

    def process(self, payload):
        """
        Does the heavy lifting.
        
        TODO: pool sentences together to optimize batch size.
        """
        
        docs = {doc['ID']: Document.from_dict(doc) for doc in payload}
        
        print(docs)
        
        for doc in docs.values():
            doc.annotate([self.endlines, self.sections, self.sentenceSplitter])
        
        sentences = []
        for doc in docs.values():
            sentences.extend(doc.get_annotations('sentence'))
            
        placeholder_doc = Document('')
        placeholder_doc.annotations = sentences
        
        placeholder_doc.annotate([self.med, self.norm])
        
        for annotation in placeholder_doc.annotations:
            if annotation.type != 'sentence':
                docs[annotation.source_ID].append(annotation)
        
        return [doc.to_dict() for doc in docs]

    async def __call__(self, request):
        """
        Adaptation to the new Ray, with async starlette request mechanism.
        """

        payload = await request.json()
        res = self.process(payload)

        return {'result': res}


@ray.remote
def put_request(docs, host = '127.0.0.1', port = '8000', endpoint = 'annotator'):
    
    docs = [x for x in docs if len(x.raw_text()) > 50]
    
    if docs == []:
        return []

    json_doc = [doc.to_dict() for doc in docs]
        
    res =  requests.post(f"http://127.0.0.1:8000/annotator", json = json_doc)
    if res.status_code != 200: 
        logger.error(res.status_code)
        logger.error(res.text)
        logger.error([x['note_id'] for x in json_doc])
        with open('notes_errors_ray.txt', 'a') as f: 
            for note in json_doc: 
                f.write(f"{note['note_id']}\n")
        return []

    res = res.json()['result']
    
    docs = [Document.from_dict(doc) for doc in res]
        
    return docs    
    
    
@timer
def process_chunk(notes, replica_chunk_size = 10, note_nlp_file = None, processed_file = None):
    
    docs = []
    
    for _, row in notes.iterrows(): 
        docs.append(Document(raw_text=row.note_text,
                            ID=row.note_id, 
                            attributes = {'person_id':row.person_id}, 
                            documentDate = row.note_datetime
        ))
    
    logger.info(f'{len(docs)} documents')
    
    try:
        res = ray.get([put_request.remote(c) for c in to_chunks(docs, replica_chunk_size)])
    except Exception as e: 
        print(f"Error: {e}")
        res = None
        
    if res is not None:
    
        flat_res  = [item for sublist in res for item in sublist]
        note_nlp = chunk_to_omop(flat_res)
    
        logger.info(f'Extracted {note_nlp.shape[0]} rows')
 
        if note_nlp_file is not None:
            dump_omop_to_csv(note_nlp, note_nlp_file)
            logger.info(f'Appended {note_nlp.shape[0]} rows to {note_nlp_file}')

        if processed_file is not None: 
            with open(processed_file, 'a') as f: 

                for ID in notes.note_id: 
                    f.write(f"{ID}\n")
    else:
        logger.error('res empty')
        return pd.DataFrame.from_records({})
        
    return note_nlp


@timer
def main_process(limit=1000, chunk_size = 5, min_date = '2020-03-01', replica_chunk_size=10, 
    note_file=None, note_nlp_file = None, processed_file = None):
    
    # engine = get_engine()

    data = pd.read_csv(note_file)
    
    print(f"{len(data)} documents to process")
    
    note_ids = data.note_id.astype(str)
   
    processed_ids = []
    if processed_file is not None: 
        processed_ids = load_processed_ids(filename=processed_file)
    
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
        
        tmp = process_chunk(data[data.note_id.isin(ids)], replica_chunk_size, note_nlp_file, processed_file)
        nrows += tmp.shape[0]
        
        time_to_current_chunk = datetime.datetime.now() - start_time
        mean_time = time_to_current_chunk / i
        ETA = mean_time * (n_chunks-i)
        logger.info(f'Processed {i}/{n_chunks} chunks. ETA: {ETA}')
        
    logger.info(f'Process done in {datetime.datetime.now()-start_time}, extracted {nrows} rows')     
    
    # engine.dispose()
    return nrows  


if __name__ == '__main__':
    
    
    logger = _get_logger()

    num_replicas = 1
    num_gpus = 1
    limit = 200
    chunk_size = 20
    replica_chunk_size= chunk_size // num_replicas
    note_file = '/export/home/bdura/pymedext/pymedext_eds/data/note.csv'
    note_nlp_file = '/export/home/bdura/pymedext/pymedext_eds/data/note_nlp.csv'
    processed_file = '/export/home/bdura/pymedext/pymedext_eds/data/processed.txt'
    min_date='2019-01-01'
    
    client = serve.start()
    
    # ray server
    config = {"num_replicas": num_replicas}
    actor_options = {"num_gpus": num_gpus}
    client.create_backend('annotator', 
                         Pipeline, 
                         'cpu',
    #                     postprocess_params,
                         config=config,
#                          ray_actor_options=actor_options
                         )
    client.create_endpoint("annotator", backend="annotator", route="/annotator", methods = ['POST'])

    
    # launch client
    main_process(limit =limit, 
                 chunk_size = chunk_size,
                 replica_chunk_size = replica_chunk_size,
                 min_date = min_date,
                 note_file=note_file,
                 note_nlp_file =note_nlp_file, 
                 processed_file = processed_file)

    
