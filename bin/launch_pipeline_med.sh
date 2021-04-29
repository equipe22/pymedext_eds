#!/bin/bash

MAIN_PATH="/export/home/edsprod/app/bigdata/pymedext-eds/run_pipeline_med.py"
CONF_PATH="/export/home/edsprod/app/bigdata/pymedext-eds/conf_pipeline_med.cf"
ENV_PATH="/export/home/edsprod/app/bigdata/env_pkg/med_env.tar.gz"

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master local[5] \
--driver-memory=10g \
--executor-memory=10g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.driver.memoryOverhead=10g \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
--archives $ENV_PATH#med_env \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./med_env/bin/python \
$MAIN_PATH $CONF_PATH
