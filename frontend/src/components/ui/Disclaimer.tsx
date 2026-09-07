import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';

export const Disclaimer: React.FC = () => {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 mb-6 relative">
      <button 
        onClick={() => setIsVisible(false)}
        className="absolute top-4 right-4 text-amber-500/60 hover:text-amber-500 transition-colors"
      >
        <X className="w-5 h-5" />
      </button>
      <div className="flex gap-3 pr-8">
        <AlertTriangle className="w-6 h-6 text-amber-500 shrink-0" />
        <div>
          <h4 className="font-semibold text-amber-500 mb-1">Medical Disclaimer</h4>
          <p className="text-sm text-amber-500/80 leading-relaxed">
            This application is a research and educational computer-aided detection system and is 
            <strong className="font-bold text-amber-400"> NOT a substitute for professional medical diagnosis</strong>. 
            Results must not be used for clinical decision-making. Always consult with a qualified healthcare provider.
          </p>
        </div>
      </div>
    </div>
  );
};
