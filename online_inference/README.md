Homework 2
==============================

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Build Docker image:
~~~
docker build -t online_inference:v1 . 
~~~

Run Docker image:
~~~
docker run -p 8000:8000  online_inference:v1
~~~

Make requests:
~~~
python model_requests.py
~~~

Test in docker root dir:
~~~
pytest
~~~