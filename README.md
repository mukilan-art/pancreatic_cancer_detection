# Explainable Deep Learning Framework for Early Pancreatic Cancer Detection

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![React](https://img.shields.io/badge/React-18.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)

## Description

This project implements an explainable deep learning pipeline designed to assist medical professionals in the early detection of pancreatic cancer from CT scans. It combines a state-of-the-art segmentation model (U-Net) to isolate the pancreas and a robust classification model (ResNet50) to predict the presence of cancerous tissues. To build trust and provide interpretability, the framework integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the specific regions in the CT scan that most strongly influenced the classification decision.

This framework is built upon the architecture and findings of recent advancements in medical imaging and aims to bridge the gap between black-box deep learning models and clinical usability.

---

## Architecture Overview

The system consists of three main pillars:
1. **Machine Learning Pipeline:** Data preprocessing, model training, and evaluation scripts built on TensorFlow and Keras.
2. **Backend API:** A high-performance FastAPI server that hosts the trained models, handles image processing, and provides predictions.
3. **Frontend Application:** A React/Vite-based web interface for users to upload CT scans and view predictions and explainability maps.

### Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | React, Vite, TailwindCSS | Fast, responsive UI for clinical users |
| **Backend** | FastAPI, Python | Asynchronous API for model inference |
| **Machine Learning** | TensorFlow, Keras | Deep learning framework for model training and deployment |
| **Image Processing** | OpenCV, NumPy, Nibabel | Handling DICOM/NIfTI and applying CLAHE enhancement |
| **Explainability** | Grad-CAM | Visualizing model attention |

---

## Folder Structure

```
pancreatic cancer detection/
├── backend/                  # FastAPI backend server
│   ├── app/                  # Application code (routers, models, logic)
│   ├── models/               # Saved model weights
│   └── main.py               # Entry point
├── frontend/                 # React web application
│   ├── src/                  # Source code
│   └── package.json          # Dependencies
├── training/                 # ML training scripts
│   ├── config.py             # Training configurations
│   ├── train_unet.py         # U-Net training script
│   ├── train_resnet50.py     # ResNet50 training script
│   └── evaluate.py           # Evaluation script
├── datasets/                 # Datasets (ignored in version control)
│   ├── pancreas-ct/          # Raw TCIA dataset
│   └── processed/            # Preprocessed images for training
├── docs/                     # Documentation
│   ├── architecture.md       # System architecture
│   ├── api.md                # API documentation
│   └── setup.md              # Setup instructions
├── README.md                 # Project README (this file)
├── .gitignore                # Git ignore rules
└── .env.example              # Environment variables template
```

---

## Installation

### Prerequisites
- Python 3.10 to 3.12 (required by the pinned TensorFlow release)
- Node.js 18 or higher
- CUDA Toolkit and cuDNN (if using GPU for inference/training)

### Backend Setup
1. Open a terminal and navigate to the project root.
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Linux/Mac: `source venv/bin/activate`
5. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Copy the environment variables template and configure it:
   ```bash
   cp ../.env.example .env
   ```

### Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```

---

## Dataset Setup

This project uses the publicly available TCIA (The Cancer Imaging Archive) Pancreas-CT dataset for training the segmentation model.

1. Create an account on TCIA if necessary.
2. Download the Pancreas-CT dataset using the NBIA Data Retriever.
3. Place the downloaded files in the `datasets/pancreas-ct/` directory.
4. (Optional) If you have a custom classification dataset, place it in `datasets/classification/` organized into `Normal/` and `Pancreatic Cancer/` subdirectories.
5. Refer to `datasets/README.md` for more details.

---

## Model Training

If you wish to train the models from scratch instead of using pre-trained weights:

### 1. Preprocess the Data
First, process the raw DICOM/NIfTI files into formatted PNGs with CLAHE enhancement.
```bash
python training/preprocess_dataset.py --data_dir datasets/pancreas-ct --output_dir datasets/processed
```

### 2. Train the U-Net Segmentation Model
This model learns to segment the pancreas from the surrounding abdominal organs.
```bash
python training/train_unet.py --epochs 100 --batch_size 4
```

### 3. Train the ResNet50 Classification Model
This model takes the masked/cropped pancreas region and classifies it.
```bash
python training/train_resnet50.py --epochs 50 --batch_size 16
```

---

## Model Weights Setup

Once training is complete (or if you have downloaded pre-trained `.h5` files):
1. Place the U-Net weights in `backend/models/unet/unet_pancreas.h5`
2. Place the ResNet50 weights in `backend/models/resnet50/resnet50_classifier.h5`

Update the paths in the `.env` file if you choose different locations.

---

## Running the Application

### Development Mode

**1. Start the Backend Server**
Ensure your virtual environment is activated.
```bash
cd backend
python -m app.main
```

**2. Start the Frontend Server**
Open a new terminal.
```bash
cd frontend
npm run dev
```
The application will be accessible at `http://localhost:5173`.

### Production Mode
For production, the React frontend should be built into static files, and the backend should be served using a production-ready WSGI/ASGI server like Gunicorn with Uvicorn workers.

**Build Frontend:**
```bash
cd frontend
npm run build
```

**Run Backend:**
```bash
cd backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## API Documentation Summary

The backend exposes a REST API for automated integration with other clinical systems.
Detailed documentation can be found in `docs/api.md` or by visiting `http://localhost:8000/docs` when the backend is running.

- `GET /health` : Health check endpoint.
- `POST /predict` : Upload a CT slice for inference. Returns classification, confidence, segmentation mask URL, and Grad-CAM URL.

---

## Evaluation

To evaluate the trained models on the test set and generate performance metrics:

```bash
python training/evaluate.py --model all
```

This script will output:
- Classification metrics (Accuracy, Precision, Recall, F1, AUC) in JSON format.
- Segmentation metrics (Dice Coefficient, IoU) in JSON format.
- Visualizations including Confusion Matrices and ROC curves saved to `training/outputs/evaluation/`.

---

## Troubleshooting

- **Memory Errors during Training (OOM):** Reduce the `batch_size` in `training/config.py` or use a smaller input resolution.
- **Model Fails to Load in Backend:** Ensure the paths in `.env` match the actual locations of your `.h5` or `.keras` files.
- **CORS Errors:** Verify that the `CORSORIGINS` variable in `.env` includes the exact URL of your frontend development server (e.g., `http://localhost:5173`).
- **NIfTI/DICOM Loading Issues:** Make sure `nibabel` and `pydicom` are installed if you are working with raw medical imaging formats.

---

## Prominent Medical Disclaimer

**IMPORTANT: FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

This software and the associated deep learning models are strictly for research and educational use. They have **NOT** been approved by the FDA, EMA, or any other regulatory body for clinical use or diagnostic purposes. 

The predictions, segmentations, and explainability maps provided by this tool should **NEVER** be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition. The authors and contributors of this project assume no liability for any use of this software.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **The Cancer Imaging Archive (TCIA):** For providing the Pancreas-CT dataset used in the segmentation training.
- **Researchers and Authors:** Inspired by recent academic papers on Explainable AI (XAI) in medical imaging, specifically regarding deep learning for early pancreatic adenocarcinoma detection.
- **Open Source Communities:** TensorFlow, Keras, React, and FastAPI communities for providing excellent tools and documentation.

---

## Detailed Model Configuration (Appendix)

### Segmentation (U-Net)
- **Input Size:** 512x512x1
- **Loss Function:** Combined Binary Cross-Entropy (BCE) and Dice Loss
- **Metrics:** Dice Coefficient, Intersection over Union (IoU)
- **Optimizer:** Adam (Initial LR: 1e-4)

### Classification (ResNet50)
- **Input Size:** 224x224x3
- **Base Model:** Pre-trained on ImageNet
- **Head:** GlobalAveragePooling2D followed by Dense layers
- **Fine-tuning:** Two-phase training (Phase 1: Freeze base, Phase 2: Unfreeze top layers)
- **Loss Function:** Categorical Cross-Entropy (or Binary)
- **Metrics:** Accuracy, Precision, Recall, AUC
