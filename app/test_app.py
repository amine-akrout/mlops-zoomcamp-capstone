import json
from fastapi.testclient import TestClient
from app import app, User

client = TestClient(app)


def test_predict_default():
    with TestClient(app) as client:
        user = User(sex=2, age=25, pay_0=2, pay_2=2, bill_amt1=50000, pay_amt1=10000)
        response = client.post("/predict", json=user.dict())
        assert response.status_code == 200
        assert response.json() == {"prediction": "Default"}


def test_predict_not_default():
    with TestClient(app) as client:
        user = User(sex=1, age=35, pay_0=0, pay_2=0, bill_amt1=20000, pay_amt1=20000)
        response = client.post("/predict", json=user.dict())
        assert response.status_code == 200
        assert response.json() == {"prediction": "Not Default"}


# test with invalid data
def test_predict_invalid_data():
    with TestClient(app) as client:
        user = User(
            sex=1,
            age=35,
            pay_0=0,
            pay_2=0,
            bill_amt1="1 million",  # Invalid data
            pay_amt1=20000,
        )
        response = client.post("/predict", json=user.dict())
        assert response.status_code == 422  # Expect a validation error


# test health endpoint
def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
