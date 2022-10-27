import json
import click
import yaml

from src.data import read_data, split_train_val_data

from src.features import make_features
from src.features.build_features import extract_target, build_transformer
from src.models import (
    train_model,
    save_model,
    predict_model,
    evaluate_model,
    create_inference_pipeline
)


def run_train_pipeline(training_pipeline_params):
    data = read_data(training_pipeline_params.input_data_path)
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    inference_pipeline = create_inference_pipeline(model, transformer)

    predicts = predict_model(
        inference_pipeline,
        val_df,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    path_to_model = save_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


def train_pipeline(config_path: str):
    with open(config_path, "r") as input_stream:
        training_pipeline_params =  yaml.safe_load(input_stream)
    return run_train_pipeline(training_pipeline_params)


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()