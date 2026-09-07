from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    # Simplified version just for segmentation
    # For now we will just call analyze and return parts of it, 
    # but a true implementation would only run segment.
    # To satisfy requirements we will reuse AnalysisService structure.
    try:
        content = await file.read()
        service = AnalysisService(settings)
        res = service.analyze(content, file.filename)
        return {
            "segmentation_mask": res.segmentation_mask,
            "segmented_image": res.segmented_image,
            "warnings": res.warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "SEGMENT_ERROR"})
