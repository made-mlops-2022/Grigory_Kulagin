import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models import train_model
from src.entities import TrainingParams


def test_train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainingParams):
    model = train_model(features, target, train_params)
    assert isinstance(model, RandomForestClassifier)