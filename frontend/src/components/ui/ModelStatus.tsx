import React from 'react';
import { ModelStatus as ModelStatusType, ComponentStatus } from '../../types';
import { CheckCircle2, AlertCircle, Clock } from 'lucide-react';
import { GlassCard } from './GlassCard';

interface ModelStatusProps {
  modelStatus: ModelStatusType | null;
  isConnected: boolean;
}

const statusSummary = (status: ComponentStatus | undefined) => {
  if (!status) {
    return { ready: false, message: 'Not loaded' };
  }

  const isReady = status.weights_loaded && status.output_valid;

  return {
    ready: isReady,
    message: isReady ? 'Ready' : 'Not loaded',
  };
};

export const ModelStatus: React.FC<ModelStatusProps> = ({ modelStatus, isConnected }) => {
  const getStatusItem = (name: string, status: ComponentStatus | undefined, warningText: string) => {
    const summary = statusSummary(status);
    const isReady = summary.ready;

    return (
      <div className="flex items-center justify-between py-2 border-b border-white/5 last:border-0 gap-3">
        <span className="text-sm text-slate-300">{name}</span>
        <div className="flex items-center gap-2">
          {isReady ? (
            <>
              <span className="text-xs text-emerald-400">{summary.message}</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            </>
          ) : (
            <>
              <span className="text-xs text-amber-400">{warningText || summary.message}</span>
              <AlertCircle className="w-4 h-4 text-amber-500" />
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <GlassCard title="System Status" icon={<Clock className="w-5 h-5" />}>
      {!isConnected ? (
        <div className="flex items-center gap-3 text-red-400 text-sm p-2 bg-red-400/10 rounded-lg">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <p>Cannot connect to the backend API. Please ensure the server is running.</p>
        </div>
      ) : (
        <div className="flex flex-col">
          {getStatusItem('Preprocessing (CLAHE)', {
            model_file_found: true,
            weights_loaded: true,
            model_trained: true,
            inference_success: true,
            output_valid: true,
          }, '')}
          {getStatusItem('U-Net Segmentation', modelStatus?.unet, 'Segmentation invalid')}
          {getStatusItem('ResNet50 Classifier', modelStatus?.resnet50, 'Classifier invalid')}
          {getStatusItem('Grad-CAM Engine', modelStatus?.gradcam, 'Grad-CAM invalid')}
        </div>
      )}
    </GlassCard>
  );
};
