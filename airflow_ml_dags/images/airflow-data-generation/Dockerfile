FROM airflow-ml-base:latest

COPY generate.py /generate.py
COPY heart_cleveland_upload.csv /heart_cleveland_upload.csv

ENTRYPOINT ["python", "generate.py"]