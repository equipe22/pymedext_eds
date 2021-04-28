ENV_PATH=/export/home/acalliger/pymedext-eds/med_env.tar.gz
DATA_PATH=/export/home/acalliger/pymedext-eds/data.tar.gz
MAIN_PATH=/export/home/acalliger/pymedext-eds/run_pipeline_med.py

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master local[2] \
--driver-memory=5g \
--executor-memory=5g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./med_env/bin/python \
--conf spark.driver.memoryOverhead=5g \
--conf "spark.executor.extraJavaOptions=-Dhdp.version=2.6.5.0-292" \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
$MAIN_PATH --input-table "coronaomop_unstable.note" --output-table "note_nlp_medoc" --write-mode "full" --process "cuda" --limit 100 --num-replica 2 --num-gpus 1 --doc-batch-size 10 --sentence_batch_size 128


