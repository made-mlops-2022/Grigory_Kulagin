import os
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.entities import TrainingParams, TrainingPipelineParams, SplittingParams
from src.models import train_model
from src.train_pipeline import run_train_pipeline


def test_train_rf_model(transformed_dataframe: Union[pd.DataFrame, pd.Series], training_params_rf: TrainingParams):
    features, target = transformed_dataframe
    model = train_model(features, target, training_params_rf)
    assert isinstance(model, RandomForestClassifier)


def test_train_log_reg_model(transformed_dataframe: Union[pd.DataFrame, pd.Series],
                             training_params_log_reg: TrainingParams):
    features, target = transformed_dataframe
    model = train_model(features, target, training_params_log_reg)
    assert isinstance(model, LogisticRegression)


def test_train_pipeline(
        fake_data_path,
        training_params_rf,
        feature_params,
        tmpdir,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    train_pipeline_params = TrainingPipelineParams(
        input_data_path=fake_data_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        feature_params=feature_params,
        train_params=training_params_rf
    )
    model_path, metrics = run_train_pipeline(train_pipeline_params)
    assert metrics["f1_score"] > 0
    assert os.path.exists(model_path)
    assert os.path.exists(train_pipeline_params.metric_path)
