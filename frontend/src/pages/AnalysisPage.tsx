import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, AlertCircle } from 'lucide-react';
import { UploadZone } from '../components/upload/UploadZone';
import { ModelStatus } from '../components/ui/ModelStatus';
import { useModelStatus } from '../hooks/useModelStatus';
import { useAnalysisContext } from '../context/AnalysisContext';
import { PipelineVisualization } from '../components/analysis/PipelineVisualization';
import { Disclaimer } from '../components/ui/Disclaimer';

export const AnalysisPage: React.FC = () => {
  const navigate = useNavigate();
  const { modelStatus, isConnected } = useModelStatus();
  const { uploadedFile, setUploadedFile } = useAnalysisContext();

  const handleFileSelect = (file: File) => {
    setUploadedFile(file);
  };

  const handleAnalyze = () => {
    if (uploadedFile) {
      navigate('/processing');
    }
  };

  const modelsLoaded = modelStatus?.unet && modelStatus?.resnet50;

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto min-h-screen">
      <Disclaimer />
      
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">New Analysis Request</h1>
        <p className="text-slate-400">Upload a CT scan to run the complete diagnostic pipeline.</p>
      </div>

      <div className="grid lg:grid-cols-12 gap-8">
        {/* Left Column - Input */}
        <div className="lg:col-span-5 space-y-6">
          <UploadZone onFileSelect={handleFileSelect} />
          
          <ModelStatus modelStatus={modelStatus} isConnected={isConnected} />
          
          {(!modelsLoaded && isConnected) && (
            <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg flex gap-3 text-amber-500 text-sm">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <p>Trained model weights are unavailable. Preprocessing can be tested, but no clinical prediction will be produced until U-Net and ResNet50 weights are installed.</p>
            </div>
          )}
        </div>

        {/* Right Column - Actions & Info */}
        <div className="lg:col-span-7 space-y-6">
          <div className="glass rounded-xl p-8 flex flex-col h-full justify-center">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">Pipeline Preview</h3>
            
            <div className="scale-90 origin-center mb-10">
              <PipelineVisualization />
            </div>

            <div className="mt-auto">
              <button
                onClick={handleAnalyze}
                disabled={!uploadedFile}
                className={`
                  w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all
                  ${uploadedFile 
                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_20px_rgba(37,99,235,0.4)] hover:shadow-[0_0_30px_rgba(37,99,235,0.6)]' 
                    : 'bg-slate-800 text-slate-500 cursor-not-allowed'}
                `}
              >
                {uploadedFile ? 'Start AI Analysis' : 'Select a File to Analyze'}
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
