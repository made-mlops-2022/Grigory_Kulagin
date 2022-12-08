import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

DF_NAME = "data.csv"
TARGET_NAME = "target.csv"

TRAIN_DATA_NAME = "train_data.csv"
TRAIN_TARGET_NAME = "train_target.csv"

TEST_DATA_NAME = "test_data.csv"
TEST_TARGET_NAME = "test_target.csv"

TEST_SIZE = 0.3


@click.command("split")
@click.argument("input_dir")
def split_data(input_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, DF_NAME))
    target = pd.read_csv(os.path.join(input_dir, TARGET_NAME))

    X_train, X_val, y_train, y_val = train_test_split(
        data,
        target,
        test_size=TEST_SIZE,
        random_state=0
    )

    X_train.to_csv(os.path.join(input_dir, TRAIN_DATA_NAME), index=False)
    y_train.to_csv(os.path.join(input_dir, TRAIN_TARGET_NAME), index=False)

    X_val.to_csv(os.path.join(input_dir, TEST_DATA_NAME), index=False)
    y_val.to_csv(os.path.join(input_dir, TEST_TARGET_NAME), index=False)


if __name__ == '__main__':
    split_data()
