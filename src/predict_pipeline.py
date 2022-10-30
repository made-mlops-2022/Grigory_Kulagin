import yaml
import click
import logging
import sys
import numpy as np
from src.data import read_data
from src.models import load_model, predict_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
s_handler = logging.StreamHandler(sys.stdout)
s_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
s_handler.setFormatter(s_format)
logger.addHandler(s_handler)


def run_predict_pipeline(predict_pipeline_params):
    test_df = read_data(predict_pipeline_params.input_data_path)
    logger.info(f'test df size = {test_df.shape[0]}')
    inference_pipeline = load_model(predict_pipeline_params.model_path)
    logger.info(f"model loaded from {predict_pipeline_params.model_path}")

    predicts = predict_model(
        inference_pipeline,
        test_df,
    )

    np.save(predict_pipeline_params.output_predicts_path, predicts)
    logger.info(f"predicts saved to {predict_pipeline_params.output_predicts_path}")

    return predicts


def predict_pipeline(config_path: str):
    with open(config_path, "r") as input_stream:
        training_pipeline_params = yaml.safe_load(input_stream)
    return run_predict_pipeline(training_pipeline_params)


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()

