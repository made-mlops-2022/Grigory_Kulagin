import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from docker.types import Mount

MODEL_PATH = Variable.get("MODEL_PATH")


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(15),
) as dag:
    predict = DockerOperator(
        image="airflow-model-prediction",
        command="/data/raw/{{ ds }}" + f" {MODEL_PATH}" + " /data/predictions/{{ ds }}",
        task_id="docker-airflow-prediction",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data",
                target="/data", type='bind'
            ),
        ]
    )

    predict
