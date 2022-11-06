from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models import train_model
from src.entities import TrainingParams


def test_train_rf_model(transformed_dataframe: Union[pd.DataFrame, pd.Series], training_params_rf: TrainingParams):
    features, target = transformed_dataframe
    model = train_model(features, target, training_params_rf)
    assert isinstance(model, RandomForestClassifier)


def test_train_log_reg_model(transformed_dataframe: Union[pd.DataFrame, pd.Series], training_params_log_reg: TrainingParams):
    features, target = transformed_dataframe
    model = train_model(features, target, training_params_log_reg)
    assert isinstance(model, LogisticRegression)


