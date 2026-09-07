from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_model_status_reports_trained_and_valid_flags():
    response = client.get('/api/models/status')
    assert response.status_code == 200
    data = response.json()
    assert 'unet' in data
    assert 'resnet50' in data
    assert 'gradcam' in data
    assert data['unet']['model_file_found'] is True
    assert data['unet']['weights_loaded'] is True
    assert data['unet']['model_trained'] is True
    assert data['resnet50']['model_file_found'] is True
    assert data['resnet50']['weights_loaded'] is True
    assert data['resnet50']['model_trained'] is True


def test_debug_response_exposes_pipeline_validation_fields():
    response = client.get('/api/diagnostics')
    assert response.status_code == 200
    data = response.json()
    assert 'gradcam_target_layer' in data
    assert 'unet_input_size' in data
    assert 'resnet50_input_size' in data
