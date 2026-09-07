from typing import Optional, Dict, List, Any
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    filename: str
    content_type: str

class ComponentState(BaseModel):
    model_file_found: bool = False
    weights_loaded: bool = False
    model_trained: bool = False
    inference_success: bool = False
    output_valid: bool = False

class ModelStatus(BaseModel):
    unet: ComponentState
    resnet50: ComponentState
    gradcam: ComponentState

class PredictionResult(BaseModel):
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None

class PreprocessingDebug(BaseModel):
    clahe_applied: bool = True

class UnetDebug(BaseModel):
    model_file_found: bool = False
    weights_loaded: bool = False
    model_trained: bool = False
    input_shape: List[int] = []
    output_shape: List[int] = []
    prediction_min: float = 0.0
    prediction_max: float = 0.0
    prediction_mean: float = 0.0
    prediction_std: float = 0.0
    mask_coverage_percent: float = 0.0
    inference_success: bool = False
    valid: bool = False

class RoiDebug(BaseModel):
    bounding_box: List[int] = []
    roi_dimensions: List[int] = []
    valid: bool = False

class Resnet50Debug(BaseModel):
    model_file_found: bool = False
    weights_loaded: bool = False
    model_trained: bool = False
    input_shape: List[int] = []
    raw_output: Optional[List[float]] = None
    normal_probability: Optional[float] = None
    cancer_probability: Optional[float] = None
    class_mapping: Dict[str, str] = {}
    inference_success: bool = False
    valid: bool = False

class GradcamDebug(BaseModel):
    target_layer: str = ""
    gradient_valid: bool = False
    heatmap_min: float = 0.0
    heatmap_max: float = 0.0
    valid: bool = False

class DebugResponse(BaseModel):
    preprocessing: PreprocessingDebug
    unet: UnetDebug
    roi: RoiDebug
    resnet50: Resnet50Debug
    gradcam: GradcamDebug

class DiagnosticsInfo(BaseModel):
    input_image_dimensions: str
    preprocessing_dimensions: str
    clahe_applied: bool
    unet_loaded_status: str
    segmentation_mask_dimensions: str
    segmentation_mask_pixel_percentage: float
    resnet50_loaded_status: str
    resnet50_input_dimensions: str
    raw_model_output: Optional[List[float]] = None
    normal_probability: Optional[float] = None
    pancreatic_cancer_probability: Optional[float] = None
    predicted_class: str
    gradcam_target_layer: str

class AnalysisResponse(BaseModel):
    id: str
    status: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    preprocessed_image: Optional[str] = None
    segmentation_mask: Optional[str] = None
    segmented_image: Optional[str] = None
    roi_image: Optional[str] = None
    gradcam_image: Optional[str] = None
    gradcam_overlay: Optional[str] = None
    processing_time: Optional[float] = None
    model_status: ModelStatus
    debug: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_status: ModelStatus

class ReportRequest(BaseModel):
    analysis_id: str
    include_images: bool
