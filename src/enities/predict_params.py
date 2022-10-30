from dataclasses import dataclass


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    output_predicts_path: str

