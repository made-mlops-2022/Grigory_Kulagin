from typing import Union
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.entities import TrainingParams


SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def save_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
