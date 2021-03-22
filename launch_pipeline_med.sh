ENV_PATH="/export/home/acalliger/pymedext-eds/env/pipeline_medicaments.tar.gz"
DATA_PATH="/export/home/acalliger/pymedext-eds/data/data.tar.gz"
CODE_PATH="/export/home/acalliger/pymedext-eds/dist/pymedext_eds-0.1.dev0-py3.6.egg"


$SPARK_HOME/bin/spark-submit \
--name medicaments \
--master yarn \
--queue default \
--deploy-mode cluster \
--num-executors=10 \
--executor-cores=5 \
--driver-memory=32g \
--executor-memory=32g \
--conf spark.default.parallelism=2000 \
--conf spark.driver.maxResultSize=32g \
--conf spark.executor.memoryOverhead=10g \
--conf spark.sql.hive.convertMetastoreOrc=false \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.sql.shuffle.partitions=2000 \
--conf spark.ui.enabled=true \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./pipeline_medicaments/bin/python \
--conf spark.yarn.appMasterEnv.TORCH_HOME=./.cache/torch \
--conf spark.executorEnv.TORCH_HOME=./.cache/torch \
--conf spark.yarn.appMasterEnv.FLAIR_CACHE_ROOT=./.flair \
--conf spark.executorEnv.FLAIR_CACHE_ROOT=./.flair \
--conf "spark.sql.autoBroadcastJoinThreshold=20000" \
--conf "spark.executor.extraJavaOptions=-Dhdp.version=2.6.5.0-292" \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
--archives $ENV_PATH#pipeline_medicaments,$DATA_PATH#data \
--py-files $CODE_PATH \
/export/home/acalliger/pymedext-eds/pipeline_med_spark.py --main_path_to_data ./data