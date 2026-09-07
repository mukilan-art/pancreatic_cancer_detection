import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisContext } from '../context/AnalysisContext';
import { PredictionCard } from '../components/results/PredictionCard';
import { ConfidenceChart } from '../components/results/ConfidenceChart';
import { ModelStatus as ModelStatusPanel } from '../components/ui/ModelStatus';
import { MetricCard } from '../components/ui/MetricCard';
import { ImageGallery } from '../components/results/ImageGallery';
import { GradCAMViewer } from '../components/results/GradCAMViewer';
import { SegmentationViewer } from '../components/results/SegmentationViewer';
import { ReportPreview } from '../components/report/ReportPreview';
import { useModelStatus } from '../hooks/useModelStatus';
import { Timer, Server } from 'lucide-react';
import { motion } from 'framer-motion';

export const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { analysisResult } = useAnalysisContext();
  const { modelStatus, isConnected } = useModelStatus();

  useEffect(() => {
    if (!analysisResult) {
      navigate('/analysis');
    }
  }, [analysisResult, navigate]);

  if (!analysisResult) return null;

  const pipelineImages = [
    { src: analysisResult.preprocessed_image, title: 'CLAHE Enhanced', subtitle: 'Contrast Limited Adaptive Histogram Equalization' },
    { src: analysisResult.segmentation_mask, title: 'Segmentation Mask', subtitle: 'U-Net output' },
    { src: analysisResult.segmented_image, title: 'Segmented Original', subtitle: 'Mask applied to enhanced image' },
    { src: analysisResult.roi_image, title: 'Region of Interest (ROI)', subtitle: 'Cropped for ResNet50 input' }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  return (
    <div className="pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto min-h-screen">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Analysis Results</h1>
          <p className="text-slate-400">Case ID: {analysisResult.id.substring(0, 8)}</p>
        </div>
        <button 
          onClick={() => navigate('/analysis')}
          className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors text-sm font-medium"
        >
          New Analysis
        </button>
      </div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="space-y-6"
      >
        <motion.div variants={itemVariants}>
          <PredictionCard 
            prediction={analysisResult.prediction}
            confidence={analysisResult.confidence}
            warnings={analysisResult.warnings}
          />
        </motion.div>

        <motion.div variants={itemVariants}>
          <ModelStatusPanel modelStatus={modelStatus} isConnected={isConnected} />
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <motion.div variants={itemVariants} className="md:col-span-2">
            <ConfidenceChart probabilities={analysisResult.probabilities} />
          </motion.div>
          <motion.div variants={itemVariants} className="space-y-6">
            <MetricCard 
              label="Processing Time" 
              value={analysisResult.processing_time?.toFixed(2) || '0.00'} 
              unit="s"
              icon={<Timer className="w-6 h-6" />}
              color="emerald"
            />
            <MetricCard 
              label="API Status" 
              value={analysisResult.status} 
              icon={<Server className="w-6 h-6" />}
              color={analysisResult.status === 'completed' ? 'emerald' : 'amber'}
            />
          </motion.div>
        </div>

        <motion.div variants={itemVariants}>
          <SegmentationViewer 
            original={analysisResult.preprocessed_image}
            mask={analysisResult.segmentation_mask}
            overlay={analysisResult.segmented_image}
          />
        </motion.div>

        <motion.div variants={itemVariants}>
          <GradCAMViewer 
            original={analysisResult.roi_image}
            heatmap={analysisResult.gradcam_image}
            overlay={analysisResult.gradcam_overlay}
          />
        </motion.div>

        <motion.div variants={itemVariants}>
          <ImageGallery images={pipelineImages.filter(img => img.src !== null)} />
        </motion.div>

        <motion.div variants={itemVariants} className="pt-10 border-t border-white/10">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Printable Report</h2>
          <ReportPreview analysis={analysisResult} />
        </motion.div>
      </motion.div>
    </div>
  );
};
