import { useState, useCallback } from 'react';
import { AnalysisResponse, ProcessingStep } from '../types';
import api from '../services/api';

const INITIAL_STEPS: ProcessingStep[] = [
  { id: 'upload', label: 'Upload', description: 'Uploading CT Scan', status: 'pending' },
  { id: 'preprocess', label: 'Preprocessing', description: 'Applying CLAHE contrast enhancement', status: 'pending' },
  { id: 'segment', label: 'Segmentation', description: 'Isolating pancreas region using U-Net', status: 'pending' },
  { id: 'classify', label: 'Classification', description: 'Analyzing for anomalies with ResNet50', status: 'pending' },
  { id: 'explain', label: 'Explainability', description: 'Generating Grad-CAM visualization', status: 'pending' }
];

export const useAnalysis = () => {
  const [file, setFileState] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>(INITIAL_STEPS);
  const [currentStep, setCurrentStep] = useState<string | null>(null);

  const setFile = useCallback((newFile: File | null) => {
    setFileState(newFile);
    if (newFile) {
      const objectUrl = URL.createObjectURL(newFile);
      setPreview(objectUrl);
    } else {
      setPreview(null);
    }
    // Reset state
    setAnalysisResult(null);
    setError(null);
    setProcessingSteps(INITIAL_STEPS);
    setCurrentStep(null);
  }, []);

  const updateStepStatus = (id: string, status: ProcessingStep['status']) => {
    setProcessingSteps(steps => 
      steps.map(step => step.id === id ? { ...step, status } : step)
    );
  };

  const startAnalysis = async (fileOverride?: File) => {
    const analysisFile = fileOverride ?? file;
    if (!analysisFile) return;

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);
    setProcessingSteps(INITIAL_STEPS.map(s => ({ ...s, status: 'pending' })));

    try {
      // Simulate sequential progress for UX since backend does it in one go
      setCurrentStep('upload');
      updateStepStatus('upload', 'processing');

      const result = await api.analyzeImage(analysisFile, (pct) => {
        if (pct === 100) {
          updateStepStatus('upload', 'completed');
          setCurrentStep('preprocess');
          updateStepStatus('preprocess', 'processing');
        }
      });

      // Simulate steps completing
      setTimeout(() => {
        updateStepStatus('preprocess', 'completed');
        setCurrentStep('segment');
        updateStepStatus('segment', 'processing');
        
        setTimeout(() => {
          updateStepStatus('segment', 'completed');
          setCurrentStep('classify');
          updateStepStatus('classify', 'processing');
          
          setTimeout(() => {
            updateStepStatus('classify', 'completed');
            setCurrentStep('explain');
            updateStepStatus('explain', 'processing');
            
            setTimeout(() => {
              updateStepStatus('explain', 'completed');
              setCurrentStep(null);
              setAnalysisResult(result);
              setIsLoading(false);
            }, 800);
          }, 1200);
        }, 1500);
      }, 1000);

    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis');
      setIsLoading(false);
      if (currentStep) {
        updateStepStatus(currentStep, 'error');
      }
    }
  };

  return {
    file,
    preview,
    analysisResult,
    isLoading,
    error,
    processingSteps,
    currentStep,
    setFile,
    startAnalysis
  };
};
