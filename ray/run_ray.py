from ray import serve
import requests

import ray

from tqdm import tqdm 

from ray.serve.utils import _get_logger
logger = _get_logger()

from ray.serve.utils import _get_logger
import time
import datetime
import pandas as pd
import math

from deploy import run_pipeline


@ray.remote
def put_request(text):
    res = requests.post(f"http://127.0.0.1:8000/annotator", json=text)
    return res.json()  


def main_process(
    limit=10, 
    note_file=None, 
):

    data = pd.read_csv(note_file)
    
    logger.info(f"{len(data)} documents to process")
    
    if limit is not None:
        data = data.iloc[:limit]
        logger.info(f"{len(data)} documents to process")
    
    # Gère la parallèlisation automatiquement
    data['results'] = ray.get([put_request.remote(text) for text in data.note_text])
    
    return data


# TODO : ajouter un main() qui
# - récupère les donnees
# - appelle le service
# - renvoie les données

# On peut aussi exécuter le processus par batch (process_chunk dans leur code)


if __name__ == '__main__':
    
    
    logger = _get_logger()

    limit = 500
    
    run_pipeline(
        num_replicas=2, 
        num_gpus=1, 
        doc_batch_size=10, 
        batch_wait_timeout=.5, 
        sentence_batch_size=128,
    )
    
    note_file = 'data/note.csv'
    note_nlp_file = 'note_nlp.csv'
    
    
    # launch client
    result = main_process(
        limit=limit, 
        note_file=note_file,
    )
    
    print(result.head())
