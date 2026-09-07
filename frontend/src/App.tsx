import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Navbar } from './components/layout/Navbar';
import { LandingPage } from './pages/LandingPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { ProcessingPage } from './pages/ProcessingPage';
import { ResultsPage } from './pages/ResultsPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { AnalysisProvider } from './context/AnalysisContext';

const PageWrapper = ({ children }: { children: React.ReactNode }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    transition={{ duration: 0.3 }}
  >
    {children}
  </motion.div>
);

function App() {
  const location = useLocation();

  return (
    <AnalysisProvider>
      <div className="min-h-screen bg-[#0a0e1a] text-slate-100 selection:bg-blue-500/30">
        <Navbar />
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<PageWrapper><LandingPage /></PageWrapper>} />
            <Route path="/analysis" element={<PageWrapper><AnalysisPage /></PageWrapper>} />
            <Route path="/processing" element={<PageWrapper><ProcessingPage /></PageWrapper>} />
            <Route path="/results" element={<PageWrapper><ResultsPage /></PageWrapper>} />
            <Route path="/model-info" element={<PageWrapper><ModelInfoPage /></PageWrapper>} />
            <Route path="/evaluation" element={<PageWrapper><EvaluationPage /></PageWrapper>} />
          </Routes>
        </AnimatePresence>
      </div>
    </AnalysisProvider>
  );
}

export default App;
