import os
import pickle

import pandas as pd
import uvicorn as uvicorn
from fastapi import FastAPI


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", 8000))