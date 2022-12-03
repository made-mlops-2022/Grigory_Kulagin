import json

import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
s_handler = logging.StreamHandler()
s_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
s_handler.setFormatter(s_format)
logger.addHandler(s_handler)


def predict():
    logger.info("Reading data from csv.")
    data = pd.read_csv("data/test_data.csv").drop('condition', axis=1).to_dict('records')
    logger.info("Data is read.")

    for request in data:
        logger.info(f"Sending request: {request}")
        response = requests.post(
            "http://localhost:8000/predict/",
            data=json.dumps(request),
        )
        logger.info(f'Status Code: {response.status_code}')
        logger.info(f'Response: {response.json()}')


if __name__ == "__main__":
    predict()
