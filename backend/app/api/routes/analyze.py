from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.schemas.analysis import AnalysisResponse
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    try:
        content = await file.read()
        service = AnalysisService(settings)
        return service.analyze(content, file.filename)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "ANALYSIS_ERROR"})
