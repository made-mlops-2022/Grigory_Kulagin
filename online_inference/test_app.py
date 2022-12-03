import json

from fastapi.testclient import TestClient
import pytest

from main import app, load_model

client = TestClient(app)

@pytest.mark.parametrize("test_request,status_code,result",
    [
        [
            {'age': 60, 'sex': 0, 'cp': 0, 'trestbps': 150, 'chol': 240, 'fbs': 0, 'restecg': 0, 'thalach': 171, 'exang': 0, 'oldpeak': 0.9, 'slope': 0, 'ca': 0, 'thal': 0},
            200,
            {'condition': 'healthy'}
        ],
        [
            {'age': 61, 'sex': 1, 'cp': 0, 'trestbps': 134, 'chol': 234, 'fbs': 0, 'restecg': 0, 'thalach': 145, 'exang': 0, 'oldpeak': 2.6, 'slope': 1, 'ca': 2, 'thal': 0},
            200,
            {'condition': 'sick'}
        ]

    ]
)
def test_predict(test_request, status_code, result):
    with TestClient(app) as client:
        response = client.post(
            '/predict',
            data=json.dumps(test_request)
        )
        assert response.status_code == status_code
        assert response.json() == result