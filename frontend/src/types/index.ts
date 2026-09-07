export interface ComponentStatus {
  model_file_found: boolean;
  weights_loaded: boolean;
  model_trained: boolean;
  inference_success: boolean;
  output_valid: boolean;
}

export interface ModelStatus {
  unet: ComponentStatus;
  resnet50: ComponentStatus;
  gradcam: ComponentStatus;
}

export interface AnalysisResponse {
  id: string;
  status: string;
  prediction: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  preprocessed_image: string | null;
  segmentation_mask: string | null;
  segmented_image: string | null;
  roi_image: string | null;
  gradcam_image: string | null;
  gradcam_overlay: string | null;
  processing_time: number | null;
  model_status: ModelStatus;
  error: string | null;
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  model_status: ModelStatus;
}

export type ProcessingStep = {
  id: string;
  label: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
};

export type AppPage = 'landing' | 'analysis' | 'processing' | 'results' | 'model-info' | 'evaluation';
