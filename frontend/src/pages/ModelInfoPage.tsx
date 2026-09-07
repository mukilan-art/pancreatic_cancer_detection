import React from 'react';
import { GlassCard } from '../components/ui/GlassCard';
import { PipelineVisualization } from '../components/analysis/PipelineVisualization';
import { Layers, Brain, Search, Sliders } from 'lucide-react';

export const ModelInfoPage: React.FC = () => {
  return (
    <div className="pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto min-h-screen">
      <div className="mb-10 text-center max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-4">Architecture & Methodology</h1>
        <p className="text-slate-400">
          PancreasAI utilizes a multi-stage deep learning pipeline combining classic computer vision techniques 
          with state-of-the-art neural networks to provide accurate and explainable results.
        </p>
      </div>

      <div className="mb-16 scale-90 md:scale-100 origin-top">
        <PipelineVisualization />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <GlassCard title="1. Preprocessing (CLAHE)" icon={<Sliders className="w-5 h-5 text-blue-400" />}>
          <h4 className="text-white font-medium mb-2">Contrast Limited Adaptive Histogram Equalization</h4>
          <p className="text-slate-400 text-sm leading-relaxed mb-4">
            Medical CT scans often suffer from low contrast between soft tissues. CLAHE improves local contrast 
            and enhances the definitions of edges in each region of the image, making the pancreas boundary 
            more distinct for the subsequent segmentation network.
          </p>
          <div className="bg-slate-900/50 p-3 rounded-lg text-xs">
            <span className="text-slate-500">Input:</span> Raw CT Slice <br/>
            <span className="text-slate-500">Output:</span> Contrast-enhanced CT Slice
          </div>
        </GlassCard>

        <GlassCard title="2. Segmentation (U-Net)" icon={<Layers className="w-5 h-5 text-emerald-400" />}>
          <h4 className="text-white font-medium mb-2">Convolutional Network for Biomedical Image Segmentation</h4>
          <p className="text-slate-400 text-sm leading-relaxed mb-4">
            U-Net automatically identifies the precise location and boundaries of the pancreas. 
            By masking out irrelevant abdominal organs and tissues, we drastically reduce the noise 
            and false positives that the classification model has to process.
          </p>
          <div className="bg-slate-900/50 p-3 rounded-lg text-xs">
            <span className="text-slate-500">Input:</span> Enhanced CT Slice <br/>
            <span className="text-slate-500">Output:</span> Binary Pancreas Mask & ROI
          </div>
        </GlassCard>

        <GlassCard title="3. Classification (ResNet50)" icon={<Brain className="w-5 h-5 text-purple-400" />}>
          <h4 className="text-white font-medium mb-2">Residual Networks for Deep Learning</h4>
          <p className="text-slate-400 text-sm leading-relaxed mb-4">
            A 50-layer deep convolutional neural network pre-trained on ImageNet and fine-tuned 
            on pancreatic cancer datasets. ResNet uses skip connections to avoid the vanishing gradient 
            problem, allowing it to learn highly complex, subtle patterns indicative of early-stage tumors.
          </p>
          <div className="bg-slate-900/50 p-3 rounded-lg text-xs">
            <span className="text-slate-500">Input:</span> Pancreas ROI <br/>
            <span className="text-slate-500">Output:</span> Class Probabilities (Normal vs Cancer)
          </div>
        </GlassCard>

        <GlassCard title="4. Explainability (Grad-CAM)" icon={<Search className="w-5 h-5 text-amber-400" />}>
          <h4 className="text-white font-medium mb-2">Gradient-weighted Class Activation Mapping</h4>
          <p className="text-slate-400 text-sm leading-relaxed mb-4">
            Black-box models are unacceptable in healthcare. Grad-CAM uses the gradients of the target concept 
            flowing into the final convolutional layer to produce a coarse localization map, highlighting 
            the important regions in the image that led to the prediction.
          </p>
          <div className="bg-slate-900/50 p-3 rounded-lg text-xs">
            <span className="text-slate-500">Input:</span> ResNet50 Activations & Gradients <br/>
            <span className="text-slate-500">Output:</span> Heatmap Overlay on ROI
          </div>
        </GlassCard>
      </div>
    </div>
  );
};
