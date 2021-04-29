#!/bin/bash

MAIN_PATH=$1
CONF_PATH=$2
ENV_PATH=$3

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master local[2] \
--driver-memory=5g \
--executor-memory=5g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.driver.memoryOverhead=5g \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
--archives $ENV_PATH#med_env \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./med_env/bin/python \
$MAIN_PATH $CONF_PATH
