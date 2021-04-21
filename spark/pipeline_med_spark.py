import argparse
import sys
import os
from glob import glob
import re
from pprint import pprint
import pandas as pd
import pkg_resources
from pyspark.sql import functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from pymedextcore.document import Document
from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.utils import rawtext_loader, SCHEMA, pymedext2omop
from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer

if __name__ == "__main__":
    """
    Meant to be ran using spark-submit.
    """

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Spark job med pipeline"
    )

    parser.add_argument(
        "--main_path_to_data",
        nargs=1,
        default="./data.tar.gz/data/",
        help="path to data"
    )
    
    # Parse arguments
    args = parser.parse_args()
    MAIN_PATH_TO_DATA = args.main_path_to_data[0]

    # Build spark session and context
    spark = SparkSession.builder \
        .appName("extract") \
        .getOrCreate()
    sql = spark.sql
    sc = SparkContext.getOrCreate()

    # Read input tables
    df_note = sql("select note_id, note_text from edsomop_qua.orbis_note").dropna(subset="note_text").limit(1000)
    # df = pd.DataFrame({'note_text': ['1000 mg de doliprane matin et soir tant que la fièvre ne baisse pas.', '200g d ibuprophene tous les 3 jours']})
    # df_note = spark.createDataFrame(df)
    
    models_param = [
        {'tagger_path':f'{MAIN_PATH_TO_DATA}/models/apmed5/entities/final-model.pt' ,
        'tag_name': 'entity_pred' },
        {'tagger_path':f'{MAIN_PATH_TO_DATA}/models/apmed5/events/final-model.pt' ,
        'tag_name': 'event_pred' },
        {'tagger_path': f'{MAIN_PATH_TO_DATA}/models/apmed5/drugblob/final-model.pt',
        'tag_name': 'drugblob_pred'}
    ]

    romedi_path = glob(f'{MAIN_PATH_TO_DATA}/romedi/*.p')[0]
    
    endlines = Endlines(["raw_text"], "clean_text", ID="endlines")
    sections = SectionSplitter(['clean_text'], "section", ID= 'sections')
    sentenceSplitter = SentenceTokenizer(["section"],"sentence", ID="sentences")
    med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=models_param, device='cpu')
    norm = MedicationNormalizer(['ENT/DRUG','ENT/CLASS'], 'normalized_mention', ID='norm', romedi_path= romedi_path)

    pipeline = [endlines, sections, sentenceSplitter, med, norm]
    
    pipeline_bc = sc.broadcast(pipeline)
    
    print("pipeline broadcast ok")
    
    @F.udf(SCHEMA)
    def medicaments(col):
        p = pipeline_bc.value
        document = Document(col)
        document.annotate(p)
        return [pymedext2omop(annotation.to_dict()) for annotation in document.get_annotations('ENT/DRUG')]
    
    df_med = df_note.withColumn('annotations', medicaments(df_note.note_text))
    df_med = df_med.withColumn("extract", F.explode(df_med.annotations))
    df_med = df_med.select("note_id", "note_text", "extract.*")
    df_med.write.parquet("./results_1000_test_gpu.parquet")
    print(df_med.count())