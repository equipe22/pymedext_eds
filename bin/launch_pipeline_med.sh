#!/bin/bash

MAIN_PATH=$1
CONF_PATH=$2
ENV_PATH=$3

source $ENV_PATH/bin/activate

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master local[2] \
--driver-memory=5g \
--executor-memory=5g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.driver.memoryOverhead=5g \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
$MAIN_PATH $CONF_PATH
