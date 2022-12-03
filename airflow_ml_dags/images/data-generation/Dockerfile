FROM airflow-ml-base:latest

COPY download.py /download.py
COPY heart_cleveland_upload.csv /heart_cleveland_upload.csv

ENTRYPOINT ["python", "download.py"]