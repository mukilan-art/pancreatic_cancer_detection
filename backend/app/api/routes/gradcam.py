from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.post("/gradcam")
async def gradcam_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    try:
        content = await file.read()
        service = AnalysisService(settings)
        res = service.analyze(content, file.filename)
        return {
            "gradcam_image": res.gradcam_image,
            "gradcam_overlay": res.gradcam_overlay,
            "warnings": res.warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "GRADCAM_ERROR"})
