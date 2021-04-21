from glob import glob
import pandas as pd
import re
from pprint import pprint
import pkg_resources

from pymedextcore.document import Document
from pymedext_eds.annotators import Endlines, SentenceTokenizer, SectionSplitter
from pymedext_eds.utils import rawtext_loader, SCHEMA, pymedext2omop
from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession


def pipeline_med(df_note, process):
    ### Load models
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
    med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=models_param,  device=process)
    norm = MedicationNormalizer(['ENT/DRUG','ENT/CLASS'], 'normalized_mention', ID='norm',romedi_path= romedi_path)

    pipeline = [endlines, sections, sentenceSplitter, med, norm]

    ### Apply algo
    def medicaments(col):
        document = Document(col)
        document.annotate(pipeline)
        l = [pymedext2omop(annotation.to_dict()) for annotation in document.get_annotations('ENT/DRUG')]
        if len(l)>0:
            return l
        else:
            return None
     
    df_note["output"] = df_note["note_text"].apply(medicaments)
    df_note = df_note.explode("output")
    df_note[['lexical_variant','start', "stop", "offset", "snippet", "term_modifiers"]]= df_note["output"].apply(pd.Series)
    
    return df_note
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detection medicaments')
    parser.add_argument('--input-table', '-i', default="edsomop.orbis_note", help='La table de notes')
    parser.add_argument('--output-table', '-o', default="edsprod.note_nlp", help='La table de sortie')
    parser.add_argument('--write-mode', '-m', default="full", help='append ou full - pour completer ou ecraser la table de sortie')
    parser.add_argument('--process', '-p', default="cpu", help='cpu ou cuda')
    
    args = parser.parse_args()
    
    #### Read data 
    spark = SparkSession.builder \
            .appName("extract") \
            .getOrCreate()
    sql = spark.sql
    sc = SparkContext.getOrCreate()
    
    if args.write_mode == "full":
        df_note = sql(f"select person_id, note_datetime, note_id, note_text from {args.input_note}").dropna(subset="note_text").toPandas()
    else:
        df_old_note = sql(f"select note_id from {args.output_table}")
        df_note = sql(f"select person_id, note_datetime, note_id, note_text from {args.input_note}")
        df_note = (
            df_note
            .dropna(subset="note_text")
            .join(df_old_note, on="note_id", how="left_anti")
            .toPandas()
        )

    df_note = pipeline_med(df_note, args.process)
    
    note_schema = (
        T.StructType([T.StructField("note_id", T.StringType(), True),
                    T.StructField("note_text", T.StringType(), True),
                    T.StructField("lexical_variant", T.StringType(), True),
                    T.StructField("start", T.DoubleType(), True),
                    T.StructField("stop", T.DoubleType(), True),
                    T.StructField("offset", T.StringType(), True),
                    T.StructField("snippet", T.StringType(), True),
                    T.StructField("term_modifiers", T.StringType(), True),
                   ])
    )
    df_note_spark = spark.createDataFrame(df_note,schema=note_schema)
    df_note_spark.write.mode('overwrite').saveAsTable(args.output_table)
    
    
    
    
