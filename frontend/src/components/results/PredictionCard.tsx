import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface PredictionCardProps {
  prediction: string | null;
  confidence: number | null;
  warnings: string[];
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, confidence, warnings }) => {
  if (!prediction) return null;

  const isNormal = prediction.toLowerCase() === 'normal';
  const colorClass = isNormal ? 'text-emerald-400' : 'text-red-500';
  const bgClass = isNormal ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-red-500/10 border-red-500/20';
  const Icon = isNormal ? ShieldCheck : AlertTriangle;

  return (
    <GlassCard className={`border-2 ${bgClass}`}>
      <div className="flex flex-col md:flex-row items-center justify-between gap-6 p-4">
        
        <div className="flex items-center gap-6">
          <div className={`w-20 h-20 rounded-full flex items-center justify-center bg-slate-900/50 shadow-inner ${colorClass}`}>
            <Icon className="w-10 h-10" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-1">AI Analysis Result</p>
            <h2 className={`text-4xl md:text-5xl font-extrabold tracking-tight ${colorClass}`}>
              {prediction}
            </h2>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center p-4 bg-slate-900/50 rounded-2xl min-w-[160px]">
          <p className="text-sm text-slate-400 mb-1">Confidence Score</p>
          <div className="text-3xl font-bold text-white flex items-baseline gap-1">
            {confidence ? (confidence * 100).toFixed(1) : '--'}
            <span className="text-lg text-slate-400">%</span>
          </div>
        </div>

      </div>

      {warnings.length > 0 && (
        <div className="mt-6 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-amber-500 mb-1">System Warnings</h4>
              <ul className="list-disc list-inside text-sm text-amber-400/80 space-y-1">
                {warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}
    </GlassCard>
  );
};
