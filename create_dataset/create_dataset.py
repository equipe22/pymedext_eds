from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

from configparser import ConfigParser, ExtendedInterpolation


if __name__ == '__main__':
    if len(sys.argv) > 1:
        conf_path = Path(sys.argv[1])
    else:
        print('ERROR: configuration file path is not defined !')
        sys.exit(1)

    config_object = ConfigParser(interpolation=ExtendedInterpolation())
    config_object.read(conf_path)

    input_schema = config_object["INPUT"]["schema"]
    input_table = config_object["INPUT"]["table"]

    limit = int(config_object["INPUT"]["limit"])

    output_schema = config_object["OUTPUT"]["schema"]
    output_table = config_object["OUTPUT"]["table"]

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sql = spark.sql

    df_old_note = sql("SELECT DISTINCT note_id FROM coronaomop_unstable.note_nlp_medoc_v2")
    df_note = sql("SELECT person_id, note_datetime, note_id, note_text, note_class_source_value FROM coronaomop_unstable.note")
    df_note = (
        df_note
        .dropna(subset="note_text")
        .join(df_old_note, on="note_id", how="left_anti")
        .select("person_id", "note_datetime", "note_id", "note_text", "note_class_source_value")
        .limit(limit)
    )
        
    df_note.write.mode("overwrite").parquet('medicaments_tmp/subset_df.parquet')
