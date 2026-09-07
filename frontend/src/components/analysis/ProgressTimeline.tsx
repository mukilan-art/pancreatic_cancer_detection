import React from 'react';
import { motion } from 'framer-motion';
import { ProcessingStep } from '../../types';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';

interface ProgressTimelineProps {
  steps: ProcessingStep[];
}

export const ProgressTimeline: React.FC<ProgressTimelineProps> = ({ steps }) => {
  return (
    <div className="w-full max-w-md mx-auto">
      <div className="relative border-l-2 border-slate-700/50 ml-4 md:ml-6 space-y-8 py-2">
        {steps.map((step, index) => {
          const isCompleted = step.status === 'completed';
          const isProcessing = step.status === 'processing';
          const isError = step.status === 'error';
          const isPending = step.status === 'pending';
          
          return (
            <motion.div 
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="relative pl-8 md:pl-10"
            >
              {/* Icon / Node */}
              <div className={`absolute -left-[17px] top-1 rounded-full bg-slate-900 border-2 ${
                  isCompleted ? 'border-emerald-500 text-emerald-500' :
                  isProcessing ? 'border-blue-500 text-blue-500' :
                  isError ? 'border-red-500 text-red-500' :
                  'border-slate-600 text-slate-600'
                }`}
              >
                {isCompleted && <CheckCircle2 className="w-7 h-7 bg-slate-900 rounded-full" />}
                {isProcessing && <Loader2 className="w-7 h-7 animate-spin bg-slate-900 rounded-full p-1" />}
                {isError && <XCircle className="w-7 h-7 bg-slate-900 rounded-full" />}
                {isPending && <Circle className="w-7 h-7 bg-slate-900 rounded-full p-1 opacity-50" />}
              </div>

              {/* Content */}
              <div>
                <h4 className={`text-lg font-medium transition-colors ${
                  isCompleted ? 'text-slate-200' :
                  isProcessing ? 'text-white' :
                  isError ? 'text-red-400' :
                  'text-slate-500'
                }`}>
                  {step.label}
                </h4>
                <p className={`text-sm mt-1 transition-colors ${
                  isCompleted ? 'text-slate-400' :
                  isProcessing ? 'text-blue-300' :
                  isError ? 'text-red-400/80' :
                  'text-slate-600'
                }`}>
                  {step.description}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};
