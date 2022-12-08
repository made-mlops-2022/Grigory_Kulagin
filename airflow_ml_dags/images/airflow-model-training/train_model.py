import os
import pickle
import yaml
import click
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_DATA_NAME = "train_data.csv"
TRAIN_TARGET_NAME = "train_target.csv"


@click.command("train")
@click.argument("feature_params_dir")
@click.argument("input_dir")
@click.argument("output_model_dir")
def train_model(feature_params_dir, input_dir, output_model_dir):
    with open(feature_params_dir, 'r') as f:
        feature_params = yaml.safe_load(f)

    X = pd.read_csv(os.path.join(input_dir, TRAIN_DATA_NAME))
    y = pd.read_csv(os.path.join(input_dir, TRAIN_TARGET_NAME))['target']

    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )

    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                categorical_pipeline,
                feature_params["categorical_features"],
            ),
            (
                "numerical_pipeline",
                num_pipeline,
                feature_params["numerical_features"],
            ),
        ]
    )

    model_pipeline = make_pipeline(transformer, LogisticRegression())
    model_pipeline.fit(X, y)

    os.makedirs(output_model_dir, exist_ok=True)
    with open(os.path.join(output_model_dir, 'model.pkl'), "wb") as f:
        pickle.dump(model_pipeline, f)


if __name__ == '__main__':
    train_model()
