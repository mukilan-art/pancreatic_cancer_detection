import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Brain, Layers, Search, ShieldAlert, Cpu, Code } from 'lucide-react';
import { PipelineVisualization } from '../components/analysis/PipelineVisualization';
import { GlassCard } from '../components/ui/GlassCard';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden min-h-[90vh] flex flex-col justify-center">
        <div className="absolute inset-0 hero-gradient opacity-50 pointer-events-none" />
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgyNTUsIDI1NSwgMjU1LCAwLjA1KSIvPjwvc3ZnPg==')] pointer-events-none" />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 text-white">
                Explainable AI for Early <br className="hidden md:block" />
                <span className="text-gradient">Pancreatic Cancer Detection</span>
              </h1>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <p className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto leading-relaxed">
                An advanced deep learning pipeline combining CLAHE enhancement, U-Net segmentation, 
                ResNet50 classification, and Grad-CAM visualizations for transparent medical imaging analysis.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <button 
                onClick={() => navigate('/analysis')}
                className="px-8 py-4 rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-bold text-lg hover:from-blue-500 hover:to-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:shadow-[0_0_30px_rgba(6,182,212,0.6)] flex items-center gap-2"
              >
                Start Analysis <ArrowRight className="w-5 h-5" />
              </button>
              <button 
                onClick={() => navigate('/model-info')}
                className="px-8 py-4 rounded-full glass hover:bg-white/10 text-white font-medium text-lg transition-all"
              >
                Learn More
              </button>
            </motion.div>
          </div>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="mt-20"
          >
            <p className="text-center text-sm font-medium text-slate-500 mb-6 uppercase tracking-widest">End-to-End Processing Pipeline</p>
            <PipelineVisualization />
          </motion.div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-20 bg-slate-900/50 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">Powered by Advanced Technologies</h2>
            <p className="text-slate-400">Built with state-of-the-art frameworks for robust performance</p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { icon: Layers, name: 'TensorFlow/Keras', desc: 'Deep learning backend' },
              { icon: Cpu, name: 'OpenCV', desc: 'Image processing' },
              { icon: Brain, name: 'FastAPI', desc: 'High-performance API' },
              { icon: Code, name: 'React + Vite', desc: 'Modern frontend' }
            ].map((tech, i) => (
              <GlassCard key={i} hoverEffect className="text-center p-6">
                <tech.icon className="w-12 h-12 mx-auto text-blue-400 mb-4" />
                <h3 className="font-semibold text-white mb-1">{tech.name}</h3>
                <p className="text-sm text-slate-400">{tech.desc}</p>
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* Highlights */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="p-8 rounded-2xl bg-gradient-to-b from-blue-500/10 to-transparent border border-blue-500/20">
              <ShieldAlert className="w-12 h-12 text-blue-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Early Detection Focus</h3>
              <p className="text-slate-400 leading-relaxed">
                Optimized specifically for identifying subtle anomalies in CT scans that may indicate early-stage pancreatic cancer, when intervention is most effective.
              </p>
            </div>
            
            <div className="p-8 rounded-2xl bg-gradient-to-b from-cyan-500/10 to-transparent border border-cyan-500/20">
              <Layers className="w-12 h-12 text-cyan-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Precision Segmentation</h3>
              <p className="text-slate-400 leading-relaxed">
                Utilizes U-Net architecture to precisely isolate the pancreas from surrounding abdominal organs, significantly reducing false positives in classification.
              </p>
            </div>
            
            <div className="p-8 rounded-2xl bg-gradient-to-b from-amber-500/10 to-transparent border border-amber-500/20">
              <Search className="w-12 h-12 text-amber-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Explainable Results</h3>
              <p className="text-slate-400 leading-relaxed">
                Grad-CAM integration provides visual heatmaps showing exactly which regions of the scan influenced the model's prediction, building trust in AI decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-white/10 text-center">
        <p className="text-sm text-slate-500 max-w-3xl mx-auto px-4">
          Disclaimer: This application is a research and educational prototype. It is NOT FDA-approved and must not be used for clinical diagnosis or decision-making.
        </p>
      </footer>
    </div>
  );
};
