import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction_endpoint():
    # Add your test cases here
    pass
