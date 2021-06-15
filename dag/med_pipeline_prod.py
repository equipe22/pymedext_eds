from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import timedelta, datetime
from airflow.utils.dates import days_ago
import os

work_path = "/export/home/edsprod/app/bigdata/pymedext-eds/"
main_path = os.path.join(work_path, 'run_pipeline_med.py')
conf_path = os.path.join(work_path, "conf_pipeline_med.cf")
run_path_match = os.path.join(work_path, 'bin/launch_pipeline_med.sh')
env_path = "/export/home/edsprod/app/bigdata/env_pkg/med_env.tar.gz"

# bash_command="ssh gpu " + run_path_match + " " + main_path + " " + conf_path + " " + env_path

default_args = {
    'owner': 'airflow',
    'catchup': False,
    'start_date': days_ago(1),
    'retries': 0,
}


dag = DAG('Detection_Medicaments', description='detection medicaments',
          catchup=False,
          schedule_interval='0 21 * * *',
          default_args=default_args,
          )

with dag:
    get_temp_data = BashOperator(
        task_id='get_temp_data',
        bash_command="ssh gpu 'cd /export/home/edsprod/app/bigdata/pymedext-eds/create_dataset/ && /export/home/edsprod/app/bigdata/pymedext-eds/create_dataset/launch_create_dataset.sh' ",
        dag=dag,
    )

    apply_med_algo = BashOperator(
        task_id='apply_med_algo',
        bash_command="ssh gpu 'cd /export/home/edsprod/app/bigdata/pymedext-eds && /export/home/edsprod/app/bigdata/pymedext-eds/bin/launch_pipeline_med.sh' ",
        dag=dag,
    )

get_temp_data >> apply_med_algo
