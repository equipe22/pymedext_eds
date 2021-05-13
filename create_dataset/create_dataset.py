from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sql = spark.sql

df_old_note = sql("select distinct note_id from coronaomop_unstable.note_nlp_medoc_v2")
df_note = sql(f"select person_id, note_datetime, note_id, note_text, note_class_source_value from coronaomop_unstable.note")
df_note = (
    df_note
    .dropna(subset="note_text")
    .join(df_old_note, on="note_id", how="left_anti")
    .select("person_id", "note_datetime", "note_id", "note_text", "note_class_source_value")
    .limit(limit)
)
    
df_note.write.parquet('subset_df.parquet')