import os
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi_health import health

from data_scheme import InputData

app = FastAPI()
model = None


@app.get("/")
def main():
    return "Hello, world!"

@app.on_event('startup')
def load_model():
    path_to_model = os.getenv('PATH_TO_MODEL')

    with open(path_to_model, 'rb') as f:
        global model
        model = pickle.load(f)

@app.post('/predict')
def predict(data: InputData):
    data_df = pd.DataFrame([data.dict()])
    y = model.predict(data_df)
    condition = 'healthy' if not y[0] else 'sick'
    return {'condition': condition}


def check_is_ready():
    return model is not None


async def success_handler(**kwargs):
    return 'Model is ready'


async def failure_handler(**kwargs):
    return 'Model is not ready'


app.add_api_route("/health",
                  health([check_is_ready],
                         success_handler=success_handler,
                         failure_handler=failure_handler,
                         success_status=200,
                         failure_status=503
                         )
                  )