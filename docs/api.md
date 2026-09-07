# API Documentation

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### 1. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Description:** Checks if the API is running.
- **Response:**
  ```json
  {
    "status": "ok",
    "version": "1.0.0"
  }
  ```

### 2. Predict Cancer
- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Uploads a CT scan image and returns the prediction, segmentation mask, and Grad-CAM heatmap.
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: The image file (PNG, JPG, DICOM).
- **Response:**
  ```json
  {
    "prediction": "Pancreatic Cancer",
    "confidence": 0.94,
    "mask_url": "/outputs/mask_123.png",
    "gradcam_url": "/outputs/gradcam_123.png"
  }
  ```
- **Error Codes:**
  - `400 Bad Request`: Invalid file format.
  - `500 Internal Server Error`: Processing failure.
