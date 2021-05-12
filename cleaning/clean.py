from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sql = spark.sql

df_medoc = sql("select * from coronaomop_unstable.note_nlp_medoc_v2")
df_medoc = df_medoc.drop_duplicates()

sql(f"USE coronaomop_unstable")
df_medoc.write.mode('overwrite').saveAsTable("note_nlp_medoc_v2")
