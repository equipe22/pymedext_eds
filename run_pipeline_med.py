from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

from ray import serve
import requests

import ray

from ray.serve.utils import _get_logger
logger = _get_logger()

from ray.serve.utils import _get_logger
import time
import datetime
import pandas as pd
import argparse

from deploy import run_pipeline


@ray.remote
def put_request(text):
    res = requests.post(f"http://127.0.0.1:8000/annotator", json=text)
    return res.json()  


def main_process(
    df_note,
    limit=10
):
    
    logger.info(f"{len(df_note)} documents to process")
    
    # Gère la parallèlisation automatiquement
    df_note['results'] = ray.get([put_request.remote(text) for text in df_note.note_text])
    
    df_note = df_note.explode("results")
    list_col = ['note_nlp_id', 'note_id', 'person_id', 'section_concept_id', 'snippet', 'offset_begin', 'offset_end', 'lexical_variant', 
                'note_nlp_concept_id', 'note_nlp_source_concept_id', 'note_nlp_source_concept_id', 'nlp_system', 'nlp_date', 'nlp_datetime', 
               'term_exists', 'term_temporal', 'term_modifiers', 'validation_level_id']
    
    df_note[list_col]= df_note["results"].apply(pd.Series)

    return df_note


if __name__ == '__main__':
    logger = _get_logger()

    if len(sys.argv) > 1:
        conf_path = Path(sys.argv[1])
    else:
        print('ERROR: configuration file path is not defined !')
        sys.exit(1)

    # Load conf file
    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(conf_path)
    input_schema = config_object["INPUT"]["schema"]
    input_table = config_object["INPUT"]["table"]
    limit = int(config_object["INPUT"]["limit"])
    
    output_schema = config_object["OUTPUT"]["schema"]
    output_table = config_object["OUTPUT"]["table"]
    write_mode = config_object["OUTPUT"]["write_mode"]
    
    process = config_object["PROCESS"]["process"]
    
    num_replica = int(config_object["RAY_PARAMETERS"]["num_replica"])
    num_gpus = int(config_object["RAY_PARAMETERS"]["num_gpus"])
    doc_batch_size = int(config_object["RAY_PARAMETERS"]["doc_batch_size"])
    sentence_batch_size = int(config_object["RAY_PARAMETERS"]["sentence_batch_size"])

    #### Read data 
    spark = SparkSession.builder \
            .appName("extract") \
            .getOrCreate()
    sql = spark.sql
    
    # list doc to keep
    list_cr = ["CRH-HOSPI", "CRH-S", "CRH-J", "CRH-CHIR", "CRH-LT-SOR", "CRH-ORTHO", "CRH-HEMATO", "CRH-PEDIA", "CRH-NEUROL", 
           "CRH-NEONAT", "CRH-NEUROP", "CRH-EVOL", "LT-ORDO", "BMI", "PRESC-AUTRE", "PRESC-MEDIC", "CR-URGE", "LT-CONS-I", 
           "LT-TYPE", "LT-CRH", "LT-ORDO", "LT-CONS-S", "LT-SOR", "LT-CRT", "LT-CONS", "LT-AUTR"]
    
    if args.write_mode == "full":
        df_note = (
            sql(f"select person_id, note_datetime, note_id, note_text from {input_schema}.{input_table}")
            .dropna(subset="note_text")
            .filter(F.col('note_class_source_value').isin(list_cr))
            .limit(limit)
            .toPandas()
        )
    elif args.write_mode == "append":
        df_old_note = sql(f"select * from {output_schema}.{output_table}")
        df_note = sql(f"select person_id, note_datetime, note_id, note_text from {input_schema}.{input_table}")
        df_note = (
            df_note
            .dropna(subset="note_text")
            .filter(F.col('note_class_source_value').isin(list_cr))
            .join(df_old_note, on="note_id", how="left_anti")
            .limit(limit)
            .toPandas()
        )
    
    run_pipeline(
        num_replicas=num_replicas, 
        num_gpus=num_gpus, 
        doc_batch_size=doc_batch_size, 
        batch_wait_timeout=.5, 
        sentence_batch_size=sentence_batch_size,
    )
    
    ### launch client
    result = main_process(df_note, limit=limit)
    
    ### write in database
    note_schema = (
        T.StructType([
            T.StructField("note_nlp_id", T.LongType(), True),
            T.StructField("note_id", T.LongType(), True),
            T.StructField("person_id", T.StringType(), True),
            T.StructField("note_text", T.StringType(), True),
            T.StructField("section_concept_id", T.StringType(), True),
            T.StructField("snippet", T.StringType(), True),
            T.StructField("offset_begin", T.DoubleType(), True),
            T.StructField("offset_begin", T.DoubleType(), True),
            T.StructField("lexical_variant", T.StringType(), True),
            T.StructField("note_nlp_concept_id", T.DoubleType(), True),
            T.StructField("note_nlp_source_concept_id", T.DoubleType(), True),
            T.StructField("nlp_system", T.StringType(), True),
            T.StructField("nlp_date", T.TimestampType(), True),
            T.StructField("nlp_datetime", T.TimestampType(), True),
            T.StructField("term_exists", T.StringType(), True),
            T.StructField("term_temporal", T.StringType(), True),
            T.StructField("term_modifiers", T.StringType(), True),
            T.StructField("validation_level_id", T.StringType(), True)
        ])
    )
    
    df_note_spark_to_add = spark.createDataFrame(result,schema=note_schema)
    
    if args.write_mode == "full":
        sql(f"USE {output_schema}")
        df_note_spark_to_add.write.mode('overwrite').saveAsTable(output_table)
    elif args.write_mode == "append":
        sql(f"USE {output_schema}")
        df_note_nlp_all = df_old_note.union(df_note_spark_to_add)
        df_note_nlp_all.write.mode('overwrite').saveAsTable(output_table)