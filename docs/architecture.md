# System Architecture

## Overview
This system uses a modular architecture combining a FastAPI backend, a React frontend, and a TensorFlow/Keras machine learning pipeline for pancreatic cancer detection.

## Components

### Frontend (React/Vite)
- **UI Components:** React components styled with modern CSS.
- **State Management:** React hooks for managing upload states and viewing results.
- **API Client:** Axios or Fetch for communicating with the FastAPI backend.

### Backend (FastAPI)
- **API Endpoints:** RESTful endpoints for image upload, processing, and retrieving results.
- **Model Server:** Loads and serves pre-trained ResNet50 and U-Net models.
- **Explainability Module:** Generates Grad-CAM heatmaps for classification results.

### ML Pipeline (TensorFlow)
- **Data Pipeline:** Scripts for loading, augmenting, and preprocessing CT scans (CLAHE, resizing).
- **Models:** 
  - U-Net for pancreas segmentation.
  - ResNet50 for cancer classification.
- **Evaluation:** Scripts to compute metrics like Dice, IoU, Accuracy, and AUC.

## Data Flow
1. **Upload:** User uploads a CT scan slice (PNG/DICOM) via the frontend.
2. **Preprocessing:** Backend receives the image, applies CLAHE enhancement, and resizes it.
3. **Segmentation:** U-Net predicts the pancreas mask.
4. **Classification:** ResNet50 classifies the cropped/masked image as Normal or Cancer.
5. **Explainability:** Grad-CAM generates a heatmap highlighting the regions that influenced the classification.
6. **Response:** The backend returns the classification result, probability, segmented mask, and Grad-CAM image to the frontend.
