from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
import os

work_path = "/export/home/edsprod/app/bigdata/pymedext-eds/"
main_path = os.path.join(work_path, 'run_pipeline_med.py')
conf_path = os.path.join(work_path, "conf_pipeline_med.cf")
run_path_match = os.path.join(work_path, 'bin/launch_pipeline_med.sh')
env_path = "/export/home/edsprod/app/bigdata/env_pkg/med_env.tar.gz"

default_args = {
    'owner': 'airflow',
    'catchup': False,
    'start_date': days_ago(2),
    'retries': 0,
}


dag = DAG('Detection_Medicaments', description='detection medicaments',
          catchup=False,
          schedule_interval='@once',
          default_args=default_args,
          )

with dag:
    match_sivic = BashOperator(
        task_id='detect_med',
        bash_command="ssh gpu " + run_path_match + " " + main_path + " " + conf_path + " " + env_path,
        dag=dag,
    )

