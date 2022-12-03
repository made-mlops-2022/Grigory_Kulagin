import os

import click
import pandas as pd

TRAIN_DF_NAME = "data.csv"
TARGET_DF_NAME = "target.csv"


@click.command("download")
@click.argument("input_dir")
@click.argument("output_dir")
def prepare(input_dir: str, output_dir: str):

    X_raw = pd.read_csv(os.path.join(input_dir, TRAIN_DF_NAME))
    y_raw = pd.read_csv(os.path.join(input_dir, TARGET_DF_NAME))

    # here should be some data preparation
    # bun in case of our data we dont need them

    os.makedirs(output_dir, exist_ok=True)

    X_raw.to_csv(os.path.join(output_dir, TRAIN_DF_NAME), index=False)
    y_raw.to_csv(os.path.join(output_dir, TARGET_DF_NAME), index=False)


if __name__ == '__main__':
    prepare()
