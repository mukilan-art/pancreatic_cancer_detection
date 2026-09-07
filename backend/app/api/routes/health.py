from fastapi import APIRouter, Depends
from datetime import datetime
from app.schemas.analysis import HealthResponse
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)):
    service = AnalysisService(settings)
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_status=service.get_model_status()
    )
