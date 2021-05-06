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

from typing import List

from ray.serve.utils import _get_logger
import time
import datetime
import pandas as pd
import math

import torch

from pymedextcore.document import Document

from signal import pause

from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer
import pkg_resources
from glob import glob


class Pipeline:
    """
    Pipeline designed to work with Ray 1.2.0
    """
    
    def __init__(self, device=None, mini_batch_size=128):

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
                'tagger_path': '/export/home/edsprod/app/bigdata/pymedext-eds/data/models/apmed5/entities/final-model.pt',
                'tag_name': 'entity_pred'
            },
            {
                'tagger_path': '/export/home/edsprod/app/bigdata/pymedext-eds/data/models/apmed5/events/final-model.pt',
                'tag_name': 'event_pred'
            },
            {
                'tagger_path': "/export/home/edsprod/app/bigdata/pymedext-eds/data/models/apmed5/drugblob/final-model.pt",
                'tag_name': 'drugblob_pred'
            },
        ]

        self.med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=self.models_param, device=device)

        data_path = pkg_resources.resource_filename('pymedext_eds', 'data/romedi')
        romedi_path = glob(data_path + '/*.p')[0]

        self.norm = MedicationNormalizer(['ENT/DRUG', 'ENT/CLASS'], 'normalized_mention', ID='norm',
                                         romedi_path=romedi_path)

        self.pipeline = [self.endlines, self.sections, self.sentenceSplitter, self.med, self.norm]
        
    @staticmethod
    def doc2omop(annotated_doc, new_norm=False):

        annots = [x.to_dict() for x in
                  annotated_doc.get_annotations('ENT/DRUG') + annotated_doc.get_annotations('ENT/CLASS')]
        sentences = [x.to_dict() for x in annotated_doc.get_annotations('sentence')]
        
        if annots == []:
            return []
        
       
        note_id = annotated_doc.source_ID
        if annotated_doc.attributes != None:
            person_id = annotated_doc.attributes.get('person_id')
        else:
            person_id = None
        
        res = []

        for drug in annots:

            section = drug["attributes"]['section']
            sentence = [x['value'] for x in sentences if x['ID'] == drug['source_ID']][0]

            norm = None
            if 'normalized_mention' in drug["attributes"].keys():
                if (drug['attributes']['normalized_mention'] != []) & (drug['attributes']['normalized_mention'] != {}):
                    if drug['type'] == 'ENT/DRUG':
                        norm = drug['attributes']['normalized_mention']['ATC7']
                    else:
                        norm = drug['attributes']['normalized_mention']

                    # new normalization with concept code separated from label 
                    # deactivated by default to ensure retro-compatibility
                    if norm is not None and new_norm:
                        norm_extract = re.search('^([A-Z0-9]+) \((.+)\)', norm)
                        if norm_extract:
                            norm = norm_extract.group(1)
                            drug['attributes']['ATC_LABEL'] = norm_extract.group(2)
            dose = None
            dose_norm = None
            route = None
            duration = None
            duration_norm = None
            freq = None
            freq_norm = None

            modifiers = []
            if drug['attributes'] != {}:
                for att, val in drug['attributes'].items():

                    if att in ['ENT/ROUTE', 'ENT/DOSE', 'ENT/DURATION', 'ENT/FREQ', 'ENT/CONDITION']:
                        for v in val:
                            modifiers.append(f"{att}='{v['value']}'")
                            if 'normalized_value' in v.keys():
                                modifiers.append(f"{att}_norm='{v['normalized_mention']}'")

                    elif att != "normalized_mention":
                        modifiers.append(f"{att}='{val}'")

            note_nlp_item = {
                'note_nlp_id': None,
                'section_concept_id': section,
                'snippet': sentence,
                'offset_begin': drug['span'][0],
                'offset_end': drug['span'][1],
                'lexical_variant': drug['value'],
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

    def process(self, documents: List[str]):
        """
        Does the heavy lifting.
        
        TODO: pool sentences together to optimize batch size.
        """
        
        docs = [Document(doc) for doc in documents]
        logger.info(f"len doc 0: {len(documents[0])}")
        
        for doc in docs:
            doc.annotate([self.endlines, self.sections, self.sentenceSplitter])
        
        sentences = []
        for i, doc in enumerate(docs):
            for annotation in doc.get_annotations('sentence'):
                annotation.attributes['doc_id'] = i
                sentences.append(annotation)
        
        logger.info(f"Processing {len(docs)} documents, {len(sentences)} sentences")
            
        placeholder_doc = Document('')
        placeholder_doc.annotations = sentences
        
        with torch.no_grad():
            placeholder_doc.annotate([self.med, self.norm])
            
        logger.info(f"lg placeholder : {placeholder_doc.get_annotations('ENT/DRUG')}")
        for annotation in placeholder_doc.get_annotations('ENT/DRUG'):
            i = annotation.attributes['doc_id']
            del annotation.attributes['doc_id']
            
            docs[i].annotations.append(annotation)
        
        return [self.doc2omop(doc) for doc in docs]
    
    @serve.accept_batch
    async def __call__(self, requests: List):
        """
        Adaptation to Ray 1.2, with async starlette request mechanism.
        """
        
        documents = []
        
        for request in requests:
            payload = await request.json()
            documents.append(payload)
        
        res = self.process(documents)

        return res


def run_pipeline(num_replicas=1, num_gpus=1, doc_batch_size=10, batch_wait_timeout=.5, sentence_batch_size=128):
    
    client = serve.start()
    
    config = dict(
        num_replicas=num_replicas,
        max_batch_size=doc_batch_size,
        batch_wait_timeout=batch_wait_timeout,
    )
    
    # ray server
    actor_options = {"num_gpus": num_gpus}
    
    client.create_backend(
        'annotator', 
        Pipeline, 
        'cuda',
        sentence_batch_size,
        config=config,
        ray_actor_options=actor_options,
    )
    
    client.create_endpoint(
        "annotator", 
        backend="annotator", 
        route="/annotator", 
        methods=['POST'],
    )


if __name__ == '__main__':
    
    logger = _get_logger()
    
    run_pipeline(
        num_replicas=1, 
        num_gpus=1, 
        doc_batch_size=10, 
        batch_wait_timeout=.5, 
        sentence_batch_size=128,
    )
    
    try:
        pause()
    except KeyboardInterrupt:
        logger.info('Quitting...')

