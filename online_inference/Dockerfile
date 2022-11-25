FROM python:3.9.13-slim-buster
RUN python -m pip install --upgrade pip

COPY models/RF_model.pkl main.py requirements.txt data_scheme.py ./

WORKDIR .

RUN pip install --no-cache-dir -r requirements.txt

ENV PATH_TO_MODEL="RF_model.pkl"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]