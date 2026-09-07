import axios from 'axios';
import { AnalysisResponse, HealthResponse, ModelStatus } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_URL,
});

export const checkHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || 'Failed to connect to backend server');
  }
};

export const analyzeImage = async (
  file: File,
  onUploadProgress?: (pct: number) => void
): Promise<AnalysisResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<AnalysisResponse>('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onUploadProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onUploadProgress(percentCompleted);
        }
      },
    });

    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || 'Analysis failed');
  }
};

export const getModelStatus = async (): Promise<ModelStatus> => {
  const health = await checkHealth();
  return health.model_status;
};

const api = {
  checkHealth,
  analyzeImage,
  getModelStatus,
};

export default api;
