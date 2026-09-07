import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AnalysisResponse } from '../types';

interface AnalysisContextType {
  analysisResult: AnalysisResponse | null;
  setAnalysisResult: (result: AnalysisResponse | null) => void;
  uploadedFile: File | null;
  setUploadedFile: (file: File | null) => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  return (
    <AnalysisContext.Provider value={{ analysisResult, setAnalysisResult, uploadedFile, setUploadedFile }}>
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysisContext = () => {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error('useAnalysisContext must be used within an AnalysisProvider');
  }
  return context;
};
