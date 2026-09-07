from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService
from app.utils.image_utils import load_image_from_bytes, numpy_to_base64, validate_file

router = APIRouter()

@router.post("/preprocess")
async def preprocess_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    try:
        content = await file.read()
        validate_file(file.filename, content, settings.MAX_UPLOAD_SIZE)
        image = load_image_from_bytes(content, file.filename)
        
        service = AnalysisService(settings)
        preprocessed = service.clahe_processor.preprocess(image)
        
        return {
            "original": numpy_to_base64(preprocessed["original"]),
            "denoised": numpy_to_base64(preprocessed["denoised"]),
            "clahe_enhanced": numpy_to_base64(preprocessed["clahe_enhanced"]),
            "resized_224": numpy_to_base64(preprocessed["resized_224"]),
            "resized_512": numpy_to_base64(preprocessed["resized_512"]),
            "warnings": []
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "PREPROCESS_ERROR"})
