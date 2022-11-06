from typing import Tuple, List

import pandas as pd
import pytest
from faker import Faker

from src.entities import FeatureParams, TrainingParams
from src.features.build_features import extract_target, build_transformer, make_features


@pytest.fixture()
def target_col() -> str:
    return "condition"


@pytest.fixture()
def numerical_features():
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture()
def categorical_features():
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


def create_fake_data(n_rows) -> pd.DataFrame:
    Faker.seed(42)
    faker = Faker()

    data = {
        "age": [faker.pyint(min_value=18, max_value=100) for _ in range(n_rows)],
        "sex": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "cp": [faker.pyint(min_value=0, max_value=3) for _ in range(n_rows)],
        "trestbps": [faker.pyint(min_value=100, max_value=200) for _ in range(n_rows)],
        "chol": [faker.pyint(min_value=100, max_value=550) for _ in range(n_rows)],
        "fbs": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "restecg": [faker.pyint(min_value=0, max_value=2) for _ in range(n_rows)],
        "thalach": [faker.pyint(min_value=70, max_value=200) for _ in range(n_rows)],
        "exang": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "oldpeak": [faker.pyfloat(min_value=0, max_value=6.2) for _ in range(n_rows)],
        "slope": [faker.pyint(min_value=0, max_value=2) for _ in range(n_rows)],
        "ca": [faker.pyint(min_value=0, max_value=3) for _ in range(n_rows)],
        "thal": [faker.pyint(min_value=0, max_value=2) for _ in range(n_rows)],
        "condition": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)]
    }
    return pd.DataFrame(data)


@pytest.fixture()
def fake_data():
    return create_fake_data(300)


@pytest.fixture()
def fake_data_path():
    data_path = 'tests/fake_data.csv'
    create_fake_data(300).to_csv(data_path, index=False)
    return data_path


@pytest.fixture()
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str
) -> FeatureParams:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col
    )
    return feature_params


@pytest.fixture()
def training_params_rf() -> TrainingParams:
    params = TrainingParams(
        model_type="RandomForestClassifier",
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    return params


@pytest.fixture()
def training_params_log_reg() -> TrainingParams:
    params = TrainingParams(
        model_type="LogisticRegression",
        random_state=42
    )
    return params


@pytest.fixture()
def transformed_dataframe(
        fake_data: pd.DataFrame,
        feature_params: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    target = extract_target(fake_data, feature_params)
    df = fake_data.drop(feature_params.target_col, 1)
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    features = make_features(transformer, df)

    return features, target
