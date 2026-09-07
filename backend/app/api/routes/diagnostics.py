from fastapi import APIRouter, Depends
from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.get("/diagnostics")
async def get_diagnostics(settings: Settings = Depends(get_settings)):
    service = AnalysisService(settings)
    status = service.get_model_status()
    return {
        "status": "ok",
        "unet": status.unet.model_dump(),
        "resnet50": status.resnet50.model_dump(),
        "gradcam": status.gradcam.model_dump(),
        "gradcam_target_layer": service.gradcam.last_conv_layer_name if service.gradcam else "conv5_block3_out",
        "unet_input_size": "512 x 512 x 1",
        "resnet50_input_size": "224 x 224 x 3",
        "classes": settings.CLASSES
    }

@router.get("/debug")
async def get_debug_response(settings: Settings = Depends(get_settings)):
    service = AnalysisService(settings)
    return service.get_model_status().model_dump()
