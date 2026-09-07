# Setup Guide

## Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA Toolkit & cuDNN (for GPU acceleration)

## Backend Setup
1. Navigate to `backend/`.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure it.
6. Run the server: `python -m app.main`

The server uses `BACKEND_HOST`, `BACKEND_PORT`, and `DEBUG` from `backend/.env`.
For example, set `BACKEND_PORT=8001` to use port 8001.

## Frontend Setup
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

## Model Weights
1. Train the models using the scripts in `training/` or download pre-trained weights.
2. Place the weights in `backend/models/unet/` and `backend/models/resnet50/` respectively.

## Troubleshooting
- **CUDA errors:** Ensure your NVIDIA drivers and CUDA toolkit versions match the TensorFlow requirements.
- **CORS issues:** Verify the `CORSORIGINS` setting in the `.env` file matches your frontend URL.
