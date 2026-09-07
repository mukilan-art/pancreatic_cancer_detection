import React from 'react';
import { GlassCard } from '../ui/GlassCard';
import { Search } from 'lucide-react';
import { ImageViewer } from './ImageViewer';

interface GradCAMViewerProps {
  original: string | null;
  heatmap: string | null;
  overlay: string | null;
}

export const GradCAMViewer: React.FC<GradCAMViewerProps> = ({ original, heatmap, overlay }) => {
  return (
    <GlassCard title="Explainability Analysis (Grad-CAM)" icon={<Search className="w-5 h-5" />}>
      <div className="mb-6">
        <p className="text-sm text-slate-300">
          Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes the regions of the CT scan that most strongly influenced the model's prediction. 
          <span className="text-amber-400 font-medium ml-1">Warmer colors (red/yellow) indicate higher importance.</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <ImageViewer 
          src={original} 
          title="Original ROI" 
          subtitle="Cropped region from U-Net"
        />
        <ImageViewer 
          src={heatmap} 
          title="Activation Heatmap" 
          subtitle="Raw Grad-CAM output"
        />
        <div className="flex flex-col h-full">
          <ImageViewer 
            src={overlay} 
            title="Overlay Visualization" 
            subtitle="Heatmap superimposed on original"
            className="flex-1"
          />
        </div>
      </div>
    </GlassCard>
  );
};
