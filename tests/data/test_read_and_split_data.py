import pandas as pd
from src.data.make_dataset import read_data, split_train_val_data
from src.entities import SplittingParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert isinstance(data, pd.DataFrame)
    assert target_col in data.columns
    assert len(data) > 10


def test_split_dataset(data: pd.DataFrame):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=42, val_size=val_size)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] >= int(data.shape[0] * (1 - val_size))
    assert val.shape[0] >= int(data.shape[0] * val_size)