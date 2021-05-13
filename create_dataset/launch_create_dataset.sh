#!/bin/bash

MAIN_PATH="/export/home/edsprod/app/bigdata/pymedext-eds/create_dataset/create_dataset.py"
ENV_PATH="/export/home/edsprod/med_env"

source $ENV_PATH/bin/activate

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master yarn \
--deploy-mode cluster \
--num-executors 6 \
--executor-cores 5 \
--driver-memory=20g \
--executor-memory=20g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.driver.memoryOverhead=10g \
--conf spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation=true \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
$MAIN_PATH
