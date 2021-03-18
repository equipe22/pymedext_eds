import context

from glob import glob
import pandas as pd
import re
from pprint import pprint
import pkg_resources

from pymedextcore.document import Document
from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.utils import rawtext_loader
from pymedext_eds.med import MedicationAnnotator, NewMedicationAnnotator, MedicationNormalizer

import pytest


@pytest.fixture
def pipeline():
    
    models_param = [
        {'tagger_path':'data/models/apmed5/entities/final-model.pt' ,
        'tag_name': 'entity_pred' },
        {'tagger_path':'data/models/apmed5/events/final-model.pt' ,
        'tag_name': 'event_pred' },
        {'tagger_path': "data/models/apmed5/drugblob/final-model.pt",
        'tag_name': 'drugblob_pred'}
    ]

    data_path = pkg_resources.resource_filename('pymedext_eds', 'data/romedi')
    romedi_path = glob(data_path + '/*.p')[0]
    
    endlines = Endlines(["raw_text"], "clean_text", ID="endlines")
    sections = SectionSplitter(['clean_text'], "section", ID= 'sections')
    sentenceSplitter = SentenceTokenizer(["section"],"sentence", ID="sentences")
    med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=models_param,  device='cpu')
    norm = MedicationNormalizer(['ENT/DRUG','ENT/CLASS'], 'normalized_mention', ID='norm',romedi_path= romedi_path)
    
    pipeline = [endlines, sections, sentenceSplitter, med, norm]
    
    return pipeline

def test_long_sentence(pipeline):
    
    doc = Document("doliprane 3 fois par jour tant que la fièvre ne basse pas et " * 100)
    doc.annotate(pipeline)
    doc.annotations[-1].to_dict()