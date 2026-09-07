import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { GlassCard } from './GlassCard';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ 
  title = 'An Error Occurred', 
  message, 
  onRetry 
}) => {
  return (
    <GlassCard className="border-red-500/20 bg-red-500/5 max-w-lg mx-auto w-full">
      <div className="flex flex-col items-center text-center p-6">
        <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
          <AlertCircle className="w-6 h-6 text-red-400" />
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-slate-300 text-sm mb-6">{message}</p>
        
        {onRetry && (
          <button 
            onClick={onRetry}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors text-sm font-medium"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
        )}
      </div>
    </GlassCard>
  );
};
