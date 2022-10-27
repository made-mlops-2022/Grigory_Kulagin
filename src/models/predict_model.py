import numpy as np
import pandas as pd
from typing import Dict, Union

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:

    return {
        "accuracy": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnClassifierModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])