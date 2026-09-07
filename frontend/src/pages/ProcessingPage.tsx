import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisContext } from '../context/AnalysisContext';
import { useAnalysis } from '../hooks/useAnalysis';
import { ProgressTimeline } from '../components/analysis/ProgressTimeline';
import { GlassCard } from '../components/ui/GlassCard';
import { Clock } from 'lucide-react';

export const ProcessingPage: React.FC = () => {
  const navigate = useNavigate();
  const { uploadedFile, setAnalysisResult } = useAnalysisContext();
  const { setFile, startAnalysis, processingSteps, analysisResult, error } = useAnalysis();
  const [elapsedTime, setElapsedTime] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const startedAt = useRef<number | null>(null);

  useEffect(() => {
    if (!uploadedFile) {
      navigate('/analysis');
      return;
    }

    if (!hasStarted) {
      setFile(uploadedFile);
      startedAt.current = Date.now();
      setElapsedTime(0);
      startAnalysis(uploadedFile);
      setHasStarted(true);
    }
  }, [uploadedFile, setFile, startAnalysis, navigate, hasStarted]);

  useEffect(() => {
    if (analysisResult) {
      setAnalysisResult(analysisResult);
      // Brief delay before redirect to show 100% completion
      const timer = setTimeout(() => {
        navigate('/results');
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [analysisResult, navigate, setAnalysisResult]);

  useEffect(() => {
    if (!hasStarted || analysisResult || error || startedAt.current === null) return;

    const updateElapsedTime = () => {
      setElapsedTime(Math.floor((Date.now() - startedAt.current!) / 1000));
    };
    updateElapsedTime();
    const interval = setInterval(updateElapsedTime, 250);
    return () => clearInterval(interval);
  }, [hasStarted, analysisResult, error]);

  const handleCancel = () => {
    navigate('/analysis');
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="pt-32 pb-12 px-4 flex flex-col items-center justify-center min-h-screen">
      <div className="w-full max-w-2xl">
        <GlassCard className="p-8">
          <div className="text-center mb-10">
            <h2 className="text-2xl font-bold text-white mb-2">Processing CT Scan</h2>
            <p className="text-slate-400">Running Explainable AI Pipeline</p>
            
            <div className="mt-4 flex items-center justify-center gap-2 text-blue-400 font-mono bg-blue-500/10 w-fit mx-auto px-4 py-2 rounded-lg">
              <Clock className="w-4 h-4" />
              <span>{formatTime(elapsedTime)}</span>
            </div>
          </div>

          <div className="mb-10 pl-4 md:pl-12">
            <ProgressTimeline steps={processingSteps} />
          </div>

          {error && (
            <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-center">
              <p className="font-semibold mb-1">Analysis Failed</p>
              <p className="text-sm">{error}</p>
            </div>
          )}

          <div className="flex justify-center">
            <button 
              onClick={handleCancel}
              className="px-6 py-2 rounded-lg border border-slate-600 hover:bg-white/5 text-slate-300 transition-colors"
            >
              Cancel Process
            </button>
          </div>
        </GlassCard>
      </div>
    </div>
  );
};
