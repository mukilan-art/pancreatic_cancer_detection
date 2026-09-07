import React from 'react';
import { motion } from 'framer-motion';
import { FileImage, Sliders, Layers, Brain, Search, FileText, ArrowRight } from 'lucide-react';

export const PipelineVisualization: React.FC = () => {
  const steps = [
    { id: 'input', label: 'CT Scan', icon: FileImage, color: 'text-slate-400', bg: 'bg-slate-800' },
    { id: 'clahe', label: 'CLAHE', icon: Sliders, color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { id: 'unet', label: 'U-Net', icon: Layers, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { id: 'resnet', label: 'ResNet50', icon: Brain, color: 'text-purple-400', bg: 'bg-purple-500/10' },
    { id: 'gradcam', label: 'Grad-CAM', icon: Search, color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { id: 'report', label: 'Report', icon: FileText, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
  ];

  return (
    <div className="w-full py-8 overflow-x-auto">
      <div className="min-w-[800px] flex items-center justify-between px-4">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const isLast = index === steps.length - 1;
          
          return (
            <React.Fragment key={step.id}>
              <motion.div 
                className="flex flex-col items-center gap-3 relative z-10"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <div className={`w-16 h-16 rounded-2xl ${step.bg} flex items-center justify-center border border-white/5 shadow-lg backdrop-blur-sm relative group`}>
                  <div className="absolute inset-0 rounded-2xl bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  <Icon className={`w-8 h-8 ${step.color}`} />
                </div>
                <span className="text-sm font-medium text-slate-300">{step.label}</span>
              </motion.div>
              
              {!isLast && (
                <motion.div 
                  className="flex-1 flex items-center justify-center px-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.1 + 0.2, duration: 0.5 }}
                >
                  <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-slate-600 to-transparent relative">
                    <motion.div 
                      className="absolute inset-y-0 left-0 bg-blue-500"
                      initial={{ width: "0%" }}
                      animate={{ width: "100%" }}
                      transition={{ 
                        delay: index * 0.1 + 0.5, 
                        duration: 1, 
                        repeat: Infinity, 
                        repeatType: "loop",
                        repeatDelay: 2
                      }}
                    />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 shrink-0 ml-[-8px] z-10" />
                </motion.div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
