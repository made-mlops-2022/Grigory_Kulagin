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
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(30),
) as dag:
    prepare_data = DockerOperator(
        image="airflow-data-preparation",
        command="/data/raw/{{ ds }} /data/processed/{{ ds }}",
        task_id="docker-airflow-preparation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(
            source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data",
            target="/data", type='bind')]
    )

    split_data = DockerOperator(
        image="airflow-data-splitting",
        command="/data/processed/{{ ds }}",
        task_id="docker-airflow-splitting",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(
            source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data",
            target="/data", type='bind')]
    )

    train_model = DockerOperator(
        image="airflow-model-training",
        command="configs/feature_config.yaml /data/processed/{{ ds }} data/models/{{ ds }}",
        task_id="docker-airflow-training",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data",
                target="/data", type='bind'
            ),
            Mount(
                source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/configs",
                target="/configs", type='bind'
            ),
        ]
    )

    validate_model = DockerOperator(
        image="airflow-model-validation",
        command="/data/processed/{{ ds }} data/models/{{ ds }} data/metrics/{{ ds }}",
        task_id="docker-airflow-validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/Users/grigory/Yandex.Disk.localized/MADE/semestr_1/MLOps/Grigory_Kulagin/airflow_ml_dags/data",
                target="/data", type='bind'
            ),
        ]
    )

    prepare_data >> split_data >> train_model >> validate_model
