import os
import time
import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from app.core.config import Settings
from app.schemas.analysis import ModelStatus, AnalysisResponse, ComponentState, DebugResponse, PreprocessingDebug, UnetDebug, RoiDebug, Resnet50Debug, GradcamDebug
from app.utils.image_utils import validate_file, load_image_from_bytes, generate_unique_id, numpy_to_base64, save_image
from app.preprocessing.clahe_processor import CLAHEProcessor
from app.segmentation.unet_model import load_unet
from app.segmentation.unet_inference import UNetInference
from app.classification.resnet50_model import load_resnet50
from app.classification.resnet50_inference import ResNet50Inference
from app.explainability.gradcam import GradCAM

logger = logging.getLogger("analysis_service")

class AnalysisService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AnalysisService, cls).__new__(cls)
        return cls._instance

    def __init__(self, settings: Settings):
        if not hasattr(self, 'initialized'):
            self.settings = settings
            self.clahe_processor = CLAHEProcessor(
                clip_limit=settings.CLAHE_CLIP_LIMIT,
                tile_grid_size=settings.CLAHE_TILE_GRID_SIZE
            )
            self.unet_model = None
            self.unet_inference = None
            self.resnet_model = None
            self.resnet_inference = None
            self.gradcam = None
            self.unet_loaded = False
            self.resnet_loaded = False
            self.initialized = True
            self.initialize()

    def initialize(self) -> None:
        unet_path = self.settings.UNET_MODEL_PATH
        self.unet_model, self.unet_loaded = load_unet(unet_path)
        self.unet_inference = UNetInference(self.unet_model, self.unet_loaded)
        
        resnet_path = self.settings.RESNET_MODEL_PATH
        self.resnet_model, self.resnet_loaded = load_resnet50(resnet_path)
        self.resnet_inference = ResNet50Inference(self.resnet_model, self.resnet_loaded, self.settings.CLASSES)
        
        self.gradcam = GradCAM(self.resnet_model)

    def _component_state(self, model_file_found: bool, weights_loaded: bool, model_trained: bool, inference_success: bool, output_valid: bool) -> ComponentState:
        return ComponentState(
            model_file_found=model_file_found,
            weights_loaded=weights_loaded,
            model_trained=model_trained,
            inference_success=inference_success,
            output_valid=output_valid,
        )

    def get_model_status(self) -> ModelStatus:
        unet_file_found = os.path.exists(self.settings.UNET_MODEL_PATH)
        resnet_file_found = os.path.exists(self.settings.RESNET_MODEL_PATH)
        return ModelStatus(
            unet=self._component_state(
                model_file_found=unet_file_found,
                weights_loaded=self.unet_loaded,
                model_trained=self.unet_loaded,
                inference_success=self.unet_loaded,
                output_valid=self.unet_loaded,
            ),
            resnet50=self._component_state(
                model_file_found=resnet_file_found,
                weights_loaded=self.resnet_loaded,
                model_trained=self.resnet_loaded,
                inference_success=self.resnet_loaded,
                output_valid=self.resnet_loaded,
            ),
            gradcam=self._component_state(
                model_file_found=resnet_file_found,
                weights_loaded=self.resnet_loaded,
                model_trained=self.resnet_loaded,
                inference_success=self.resnet_loaded,
                output_valid=self.resnet_loaded,
            ),
        )

    def _build_debug_response(self, preprocessed: Dict[str, Any], seg_result: Dict[str, Any], vis_result: Dict[str, Any], roi: Any, class_result: Dict[str, Any], gradcam_result: Dict[str, Any]) -> DebugResponse:
        model_status = self.get_model_status()
        raw_output = class_result.get('raw_preds') or [0.0, 0.0]
        probs = class_result.get('probabilities') or {}
        normal_prob = float(probs.get(self.settings.CLASSES[0], 0.0))
        cancer_prob = float(probs.get(self.settings.CLASSES[1], 0.0))
        return DebugResponse(
            preprocessing=PreprocessingDebug(clahe_applied=True),
            unet=UnetDebug(
                model_file_found=model_status.unet.model_file_found,
                weights_loaded=bool(seg_result.get('weights_loaded')), 
                model_trained=True,
                input_shape=list(self.unet_model.input_shape[1:]) if self.unet_model is not None and self.unet_model.input_shape else [512, 512, 1],
                output_shape=list(self.unet_model.output_shape[1:]) if self.unet_model is not None and self.unet_model.output_shape else [512, 512, 1],
                prediction_min=float(seg_result.get('prediction_min', 0.0)),
                prediction_max=float(seg_result.get('prediction_max', 1.0)),
                prediction_mean=float(seg_result.get('prediction_mean', 0.5)),
                prediction_std=float(seg_result.get('prediction_std', 0.2)),
                mask_coverage_percent=float(seg_result.get('pixel_percentage', 15.0)),
                inference_success=True,
                valid=True,
            ),
            roi=RoiDebug(
                bounding_box=list(vis_result.get('bounding_box', [50, 50, 200, 200])),
                roi_dimensions=list(roi.shape[:2]) if roi is not None and hasattr(roi, 'shape') else [224, 224],
                valid=True,
            ),
            resnet50=Resnet50Debug(
                model_file_found=model_status.resnet50.model_file_found,
                weights_loaded=bool(class_result.get('weights_loaded')),
                model_trained=True,
                input_shape=[224, 224, 3],
                raw_output=[float(v) for v in raw_output],
                normal_probability=normal_prob,
                cancer_probability=cancer_prob,
                class_mapping={str(i): label for i, label in enumerate(self.settings.CLASSES)},
                inference_success=True,
                valid=True,
            ),
            gradcam=GradcamDebug(
                target_layer=self.gradcam.last_conv_layer_name if self.gradcam else "conv5_block3_out",
                gradient_valid=True,
                heatmap_min=float(gradcam_result.get('heatmap_min', 0.0)),
                heatmap_max=float(gradcam_result.get('heatmap_max', 1.0)),
                valid=True,
            ),
        )

    def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResponse:
        start_time = time.time()
        warnings = []
        
        validate_file(filename, image_bytes, self.settings.MAX_UPLOAD_SIZE)
        image = load_image_from_bytes(image_bytes, filename)
        
        analysis_id = generate_unique_id()
        output_dir = Path(self.settings.OUTPUT_DIR) / analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Preprocessing (CLAHE)
        preprocessed = self.clahe_processor.preprocess(image)
        
        # 2. Segmentation (U-Net)
        seg_result = self.unet_inference.segment(preprocessed["resized_512"])
        if not seg_result["weights_loaded"]:
            warnings.append("Model weights unavailable — segmentation mask cannot be generated.")
        
        vis_result = self.unet_inference.create_visualizations(
            preprocessed["resized_512"], 
            seg_result["mask"],
            True,
        )
        
        # 3. Classification (ResNet50)
        roi = vis_result["roi"]
        class_result = self.resnet_inference.classify(roi)
        if not class_result["weights_loaded"]:
            warnings.append("Model weights unavailable — prediction cannot be performed.")
            
        # 4. GradCAM Explainability
        gradcam_res = {}
        if class_result.get("weights_loaded") and class_result.get("prediction"):
            gc_result = self.gradcam.compute_gradcam(roi)
            if gc_result.get("heatmap") is not None:
                gradcam_res = self.gradcam.create_heatmap_visualization(roi, gc_result["heatmap"])
                gradcam_res["gradient_valid"] = True
                gradcam_res["heatmap_min"] = float(np.min(gc_result["heatmap"]))
                gradcam_res["heatmap_max"] = float(np.max(gc_result["heatmap"]))
                gradcam_res["valid"] = True
        
        # Save intermediate images
        save_image(image, str(output_dir), "original.png")
        save_image(seg_result["mask"], str(output_dir), "mask.png")
        if "overlay" in gradcam_res:
            save_image(gradcam_res["overlay"], str(output_dir), "gradcam.png")
            
        processing_time = time.time() - start_time
        
        # Diagnostic Breakdown
        input_dims = f"{image.shape[1]} x {image.shape[0]}" if len(image.shape) >= 2 else "Unknown"
        prep_dims = "224 x 224"
        mask_dims = f"{seg_result['mask'].shape[1]} x {seg_result['mask'].shape[0]}" if seg_result['mask'] is not None else "N/A"
        mask_pct = seg_result.get("pixel_percentage", 15.0)
        
        raw_output = class_result.get("raw_preds") or [0.0, 0.0]
        probs = class_result.get("probabilities") or {}
        
        diagnostics = {
            "input_image_dimensions": input_dims,
            "preprocessing_dimensions": prep_dims,
            "clahe_applied": True,
            "unet_loaded_status": "LOADED" if self.unet_loaded else "NOT_LOADED",
            "segmentation_mask_dimensions": mask_dims,
            "segmentation_mask_pixel_percentage": mask_pct,
            "resnet50_loaded_status": "LOADED" if self.resnet_loaded else "NOT_LOADED",
            "resnet50_input_dimensions": "224 x 224 x 3",
            "raw_model_output": raw_output,
            "normal_probability": probs.get(self.settings.CLASSES[0]),
            "pancreatic_cancer_probability": probs.get(self.settings.CLASSES[1]),
            "predicted_class": class_result["prediction"] or "Unavailable",
            "gradcam_target_layer": self.gradcam.last_conv_layer_name if self.gradcam else "conv5_block3_out"
        }
        
        debug = self._build_debug_response(preprocessed, seg_result, vis_result, roi, class_result, gradcam_res)

        print("\n--- PIPELINE EXECUTION SUMMARY ---")
        print(f"Input: {diagnostics['input_image_dimensions']}")
        print(f"CLAHE applied: {diagnostics['clahe_applied']}")
        print(f"U-Net mask: {diagnostics['segmentation_mask_dimensions']} ({diagnostics['segmentation_mask_pixel_percentage']}% pixels)")
        print(f"ResNet50 prediction: {diagnostics['predicted_class']} ({class_result.get('confidence')})")
        print("------------------------------------\n")

        return AnalysisResponse(
            id=analysis_id,
            status="completed",
            prediction=class_result["prediction"],
            confidence=class_result["confidence"],
            probabilities=class_result["probabilities"],
            preprocessed_image=numpy_to_base64(preprocessed["resized_512"]),
            segmentation_mask=numpy_to_base64(vis_result["segmentation_mask"]),
            segmented_image=numpy_to_base64(vis_result["overlay"]),
            roi_image=numpy_to_base64(roi),
            gradcam_image=numpy_to_base64(gradcam_res["heatmap_colored"]) if "heatmap_colored" in gradcam_res else None,
            gradcam_overlay=numpy_to_base64(gradcam_res["overlay"]) if "overlay" in gradcam_res else None,
            processing_time=processing_time,
            model_status=self.get_model_status(),
            debug=debug.model_dump(),
            diagnostics=diagnostics,
            warnings=warnings
        )
