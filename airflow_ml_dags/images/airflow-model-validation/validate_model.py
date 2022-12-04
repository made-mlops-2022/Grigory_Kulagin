import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TEST_DATA_NAME = "test_data.csv"
TEST_TARGET_NAME = "test_target.csv"


@click.command("validate")
@click.argument("input_data_dir")
@click.argument("input_model_dir")
@click.argument("output_metrics_dir")
def validate_model(input_data_dir, input_model_dir, output_metrics_dir):
    with open(os.path.join(input_model_dir, 'model.pkl'), "rb") as f:
        model = pickle.load(f)

    X = pd.read_csv(os.path.join(input_data_dir, TEST_DATA_NAME))
    y = pd.read_csv(os.path.join(input_data_dir, TEST_TARGET_NAME))['target']

    predicts = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, predicts),
        "f1_score": f1_score(y, predicts),
        "roc_auc": roc_auc_score(y, predicts),
    }

    os.makedirs(output_metrics_dir, exist_ok=True)
    with open(os.path.join(output_metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    validate_model()