import os

import click
import pandas as pd

TRAIN_DF_NAME = "data.csv"
TARGET_DF_NAME = "target.csv"


@click.command("download")
@click.argument("output_dir")
def generate(output_dir: str):
    df = pd.read_csv("heart_cleveland_upload.csv")
    sample_df = df.sample(frac=0.5)

    X = sample_df.drop('condition', axis=1)
    y = sample_df['condition'].to_frame(name='target')

    os.makedirs(output_dir, exist_ok=True)

    X.to_csv(os.path.join(output_dir, TRAIN_DF_NAME), index=False)
    y.to_csv(os.path.join(output_dir, TARGET_DF_NAME), index=False)


if __name__ == '__main__':
    generate()
