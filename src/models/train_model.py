import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]

# TODO: redo train_params
def train_model(
    features: pd.DataFrame, target: pd.Series, train_params
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


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }
