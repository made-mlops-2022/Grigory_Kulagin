import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "data_generator",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(30),
) as dag:
    generate = DockerOperator(
        image="airflow-data-generation",
        command="/data/raw/{{ ds }}",
        task_id="docker-airflow-generation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data", target="/data", type='bind')]
    )

    generate