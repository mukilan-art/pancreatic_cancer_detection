import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Pancreatic Cancer XAI"
    DEBUG: bool = True
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    UNET_MODEL_PATH: str = "models/unet/unet_pancreas.h5"
    RESNET_MODEL_PATH: str = "models/resnet50/resnet50_classifier.h5"
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: list = ["png", "jpg", "jpeg", "dcm"]
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: int = 8
    UNET_INPUT_SIZE: tuple = (512, 512)
    RESNET_INPUT_SIZE: tuple = (224, 224)
    CLASSES: list = ["Normal", "Pancreatic Cancer"]
    CORSORIGINS: str = "http://localhost:5173,http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
