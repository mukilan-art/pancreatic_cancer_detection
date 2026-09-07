import React from 'react';
import { GlassCard } from '../components/ui/GlassCard';
import { BarChart2, Activity, ShieldCheck, Target } from 'lucide-react';
import { EmptyState } from '../components/ui/EmptyState';
import { MetricCard } from '../components/ui/MetricCard';

export const EvaluationPage: React.FC = () => {
  // In a full implementation, we would fetch these from /api/evaluation
  // For now, as instructed, we show empty states / N/A, NO FAKE NUMBERS
  const hasData = false;

  return (
    <div className="pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto min-h-screen">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Model Evaluation Metrics</h1>
        <p className="text-slate-400">Performance statistics on the validation dataset.</p>
      </div>

      {!hasData ? (
        <GlassCard className="p-12 border-dashed border-2 border-slate-700 bg-transparent">
          <EmptyState 
            title="Evaluation Data Unavailable" 
            message="The model has not been evaluated yet or the evaluation metrics file is missing. Please run the evaluation script on the backend to generate performance metrics."
            icon={<BarChart2 className="w-16 h-16 opacity-50" />}
          />
        </GlassCard>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <MetricCard label="Accuracy" value="N/A" icon={<Target className="w-5 h-5" />} color="blue" />
            <MetricCard label="Sensitivity (Recall)" value="N/A" icon={<Activity className="w-5 h-5" />} color="emerald" />
            <MetricCard label="Specificity" value="N/A" icon={<ShieldCheck className="w-5 h-5" />} color="slate" />
            <MetricCard label="F1 Score" value="N/A" icon={<BarChart2 className="w-5 h-5" />} color="amber" />
          </div>
          
          {/* Placeholder for charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <GlassCard title="Confusion Matrix" className="h-[400px]">
              <div className="flex items-center justify-center h-full text-slate-500">Data Unavailable</div>
            </GlassCard>
            <GlassCard title="ROC Curve" className="h-[400px]">
              <div className="flex items-center justify-center h-full text-slate-500">Data Unavailable</div>
            </GlassCard>
          </div>
        </>
      )}
    </div>
  );
};
