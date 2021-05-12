#!/bin/bash

MAIN_PATH="/export/home/edsprod/app/bigdata/pymedext-eds/run_pipeline_med.py"
CONF_PATH="/export/home/edsprod/app/bigdata/pymedext-eds/conf_pipeline_med.cf"
ENV_PATH="/export/home/edsprod/med_env"

# if [ -d med_env ]; then 
#     echo "env found" 
# else 
#     mkdir med_env
#     tar -xzf $ENV_PATH -C med_env 
# fi;

source $ENV_PATH/bin/activate

export http_proxy=http://proxym-inter.aphp.fr:8080
export https_proxy=http://proxym-inter.aphp.fr:8080

$SPARK_HOME/bin/spark-submit \
--name pipeline_med \
--master local[10] \
--driver-memory=20g \
--executor-memory=20g \
--conf spark.sql.session.timeZone=Europe/Paris \
--conf spark.ui.enabled=true \
--conf spark.driver.memoryOverhead=10g \
--conf "spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxym-inter.aphp.fr -Dhttp.proxyPort=8080 -Dhttps.proxyHost=proxym-inter.aphp.fr -Dhttps.proxyPort=8080" \
$MAIN_PATH $CONF_PATH
