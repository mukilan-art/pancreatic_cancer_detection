from fastapi import APIRouter, Depends
from app.schemas.analysis import ModelStatus
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.get("/models/status", response_model=ModelStatus)
def get_models_status(settings: Settings = Depends(get_settings)):
    service = AnalysisService(settings)
    return service.get_model_status()
