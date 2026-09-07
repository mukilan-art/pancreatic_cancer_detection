from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    try:
        content = await file.read()
        service = AnalysisService(settings)
        res = service.analyze(content, file.filename)
        return {
            "prediction": res.prediction,
            "confidence": res.confidence,
            "probabilities": res.probabilities,
            "warnings": res.warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "CLASSIFY_ERROR"})
