import pytest
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import cv2
import io

client = TestClient(app)

def test_analyze_invalid_file():
    # Sending a text file instead of image
    response = client.post(
        "/api/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    # the endpoint should catch it and raise HTTPException which returns error format
    assert response.status_code in [400, 500]
    if response.status_code == 500:
        data = response.json()
        assert "detail" in data
        assert "error_code" in data

def test_analyze_valid_file():
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded_img = cv2.imencode('.png', img)
    assert success
    
    response = client.post(
        "/api/analyze",
        files={"file": ("test.png", encoded_img.tobytes(), "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] == "completed"
    assert "prediction" in data
    assert "model_status" in data

def test_preprocess_valid_file():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded_img = cv2.imencode('.png', img)
    assert success
    
    response = client.post(
        "/api/preprocess",
        files={"file": ("test.png", encoded_img.tobytes(), "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "original" in data
    assert "clahe_enhanced" in data
    assert "resized_512" in data
