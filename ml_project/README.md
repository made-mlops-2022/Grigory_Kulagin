Homework 1
==============================

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python src/train_pipeline.py configs/config_RF/train_config.yaml
python src/predict_pipeline.py configs/config_RF/predict_config.yaml
~~~

Test:
~~~
pytest tests/