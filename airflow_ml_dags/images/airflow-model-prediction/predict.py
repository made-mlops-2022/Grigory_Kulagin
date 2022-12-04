import os
import pickle

import click

import pandas as pd

DATA_NAME = "data.csv"
PREDS_NAME = "predictions.csv"
MODEL_NAME = 'model.pkl'


@click.command("predict")
@click.argument("input_dir")
@click.argument("model_dir")
@click.argument("preds_dir")
def predict(input_dir, model_dir, preds_dir):
    data = pd.read_csv(os.path.join(input_dir, DATA_NAME))

    with open(os.path.join(model_dir, MODEL_NAME), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(data)

    os.makedirs(preds_dir, exist_ok=True)
    preds = pd.DataFrame(predictions)
    preds.to_csv(os.path.join(preds_dir, PREDS_NAME), index=False)


if __name__ == '__main__':
    predict()
