# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: '[2.4.3] Py3'
#     language: python
#     name: pyspark-2.4.3
# ---

# %%
from glob import glob
import pandas as pd
import re
from pprint import pprint
import pkg_resources

from pymedextcore.document import Document
from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.utils import rawtext_loader
from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer

# %%
from pyspark.sql import functions as F

# %%
from toolbox.udf.pymedext import SCHEMA, pymedext2omop

# %%
# Snippet to increase executor limits
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

#Stop current Spark context
sc.stop()

#Loading pyspark
conf = SparkConf().setAppName("medicaments")

conf.set("spark.yarn.max.executor.failures", "2")
conf.set("spark.executor.memory", '16g')
conf.set("spark.driver.memory", '5g')
conf.set("spark.dynamicAllocation.minExecutors", "1")
conf.set("spark.dynamicAllocation.maxExecutors","2")
conf.set(f'spark.executorEnv.PYTORCH_TRANSFORMERS_CACHE', '.cache')
sc = SparkContext(conf=conf, batchSize=2)
sc.setLogLevel("WARN")

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sql = spark.sql

# %%
models_param = [
    {'tagger_path':'data/models/apmed5/entities/final-model.pt' ,
    'tag_name': 'entity_pred'},
    {'tagger_path':'data/models/apmed5/events/final-model.pt' ,
    'tag_name': 'event_pred'},
    {'tagger_path': "data/models/apmed5/drugblob/final-model.pt",
    'tag_name': 'drugblob_pred'}
]

data_path = pkg_resources.resource_filename('pymedext_eds', 'data/romedi')
romedi_path = glob(data_path + '/*.p')[0]

# %%
data_path = pkg_resources.resource_filename('pymedext_eds', 'data/demo')
file_list = glob(data_path + '/*.txt')

docs = [rawtext_loader(x) for x in file_list]

# %%
endlines = Endlines(["raw_text"], "clean_text", ID="endlines")
sections = SectionSplitter(['clean_text'], "section", ID= 'sections')
sentenceSplitter = SentenceTokenizer(["section"],"sentence", ID="sentences")
med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=models_param, device='cpu')
norm = MedicationNormalizer(['ENT/DRUG','ENT/CLASS'], 'normalized_mention', ID='norm', romedi_path=romedi_path)

pipeline = [endlines, sections, sentenceSplitter, med, norm]

# %%
doc = Document("1000 mg de doliprane matin et soir tant que la fièvre ne baisse pas.")

# %%
doc.annotate(pipeline)

# %%
print(pymedext2omop(doc.annotations[-1].to_dict())[-1])

# %%
pb = sc.broadcast(pipeline)


# %%
@F.udf(SCHEMA)
def medicaments(col):
#     assert False
    p = pb.value
    document = Document(col)
    document.annotate(p)
    return [pymedext2omop(annotation) for annotation in document.get_annotations('ENT/DRUG')]
#     return len(p)


# %%
df = pd.DataFrame({'note_text': ['1000 mg de doliprane matin et soir tant que la fièvre ne baisse pas.']})
# df = pd.DataFrame({'note_text': [doc.annotations[0].value for doc in docs]})

# %%
df.head()

# %%
notes = spark.createDataFrame(df)

# %%
notes = notes.cache()

# %%
notes.count()

# %%
notes.count()

# %%
out = notes.withColumn('nb_annotations', medicaments(notes.note_text)).toPandas()

# %%
out

# %%
1

# %%

# %%

# %%

# %%
# print(docs[0].annotations[0].value)

# %%
for doc in docs:
    doc.annotate(pipeline)

# %% jupyter={"outputs_hidden": true}
pd.DataFrame.from_records(MedicationAnnotator.doc_to_omop(docs[0])).T

# %% jupyter={"outputs_hidden": true}
doc.get_annotations('ENT/DRUG')


# %%

# %%

# %%

# %% [markdown]
# # Decorrelation

# %%
class Sentence(object):
    
    def __init__(self, text):
        
        self.value = text
        self.span = (0, len(text))
        self.ID = 0


# %%
med.mini_batch_size = 512

# %%
docs = ['1000 mg de doliprane matin et soir tant que la fièvre ne baisse pas.'] * 512

# %%
# %%timeit
sentences = [Sentence(doc) for doc in docs]
annotations = med.infer_flair(sentences[:20])

# %%
annotations[0]

# %%
med._postprocess_entities(sentences, annotations)[0]

# %%
