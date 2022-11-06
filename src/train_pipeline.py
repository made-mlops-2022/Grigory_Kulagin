import json
import logging
import sys

import click

from src.data import read_data, split_train_val_data
from src.entities import TrainingPipelineParams, read_training_pipeline_params
from src.features import make_features
from src.features.build_features import extract_target, build_transformer
from src.models import (
    train_model,
    save_model,
    predict_model,
    evaluate_model,
    create_inference_pipeline
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
s_handler = logging.StreamHandler(sys.stdout)
s_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
s_handler.setFormatter(s_format)
logger.addHandler(s_handler)


def run_train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    logger.info(f"start data reading")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape = {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train size = {train_df.shape[0]}, val size = {val_df.shape[0]}")

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    logger.info(f'train features shape is {train_features.shape}')
    logger.info("start model training")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    logger.info(f"model  is trained")

    inference_pipeline = create_inference_pipeline(model, transformer)

    predicts = predict_model(
        inference_pipeline,
        val_df,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
    )
    logger.info(f"model metrics is {metrics}")

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f'metrics saved to path: {training_pipeline_params.output_model_path}')

    path_to_model = save_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    logger.info(f'model saved to path: {training_pipeline_params.output_model_path}')

    return path_to_model, metrics


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
