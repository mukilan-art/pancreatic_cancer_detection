import React from 'react';
import { AnalysisResponse } from '../../types';
import { Printer, Download, FileText } from 'lucide-react';
import { GlassCard } from '../ui/GlassCard';

interface ReportPreviewProps {
  analysis: AnalysisResponse;
}

export const ReportPreview: React.FC<ReportPreviewProps> = ({ analysis }) => {
  const handlePrint = () => {
    window.print();
  };

  const isNormal = analysis.prediction?.toLowerCase() === 'normal';

  return (
    <GlassCard title="Clinical Report Preview" icon={<FileText className="w-5 h-5" />}>
      <div className="flex justify-end gap-3 mb-6">
        <button onClick={handlePrint} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors text-sm font-medium">
          <Printer className="w-4 h-4" /> Print
        </button>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors text-sm font-medium">
          <Download className="w-4 h-4" /> Download PDF
        </button>
      </div>

      <div className="bg-white text-slate-900 p-8 rounded-lg shadow-inner max-w-4xl mx-auto printable-report">
        {/* Header */}
        <div className="border-b-2 border-slate-200 pb-6 mb-6 flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 mb-1">PancreasAI Analysis Report</h1>
            <p className="text-sm text-slate-500">Explainable Deep Learning Framework for Early Pancreatic Cancer Detection</p>
          </div>
          <div className="text-right text-sm text-slate-600">
            <p><span className="font-semibold">Case ID:</span> {analysis.id.substring(0, 8).toUpperCase()}</p>
            <p><span className="font-semibold">Date:</span> {new Date().toLocaleDateString()}</p>
            <p><span className="font-semibold">Time:</span> {new Date().toLocaleTimeString()}</p>
          </div>
        </div>

        {/* Prediction Summary */}
        <div className="mb-8">
          <h2 className="text-lg font-bold uppercase tracking-wider text-slate-500 border-b border-slate-200 pb-2 mb-4">Diagnostic Summary</h2>
          <div className={`p-4 rounded-lg border-2 ${isNormal ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm font-semibold text-slate-600 uppercase mb-1">AI Prediction</p>
                <h3 className={`text-3xl font-bold ${isNormal ? 'text-green-700' : 'text-red-700'}`}>
                  {analysis.prediction}
                </h3>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-slate-600 uppercase mb-1">Confidence</p>
                <p className="text-3xl font-bold text-slate-800">
                  {analysis.confidence ? (analysis.confidence * 100).toFixed(1) : '--'}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Details */}
        <div className="mb-8">
          <h2 className="text-lg font-bold uppercase tracking-wider text-slate-500 border-b border-slate-200 pb-2 mb-4">Technical Details</h2>
          <div className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
            <div><span className="font-semibold">Processing Time:</span> {analysis.processing_time?.toFixed(2)}s</div>
            <div><span className="font-semibold">Model Status:</span> U-Net (Active), ResNet50 (Active)</div>
            
            {analysis.probabilities && (
              <div className="col-span-2 mt-4">
                <span className="font-semibold block mb-2">Class Probabilities:</span>
                <ul className="list-disc pl-5 space-y-1">
                  {Object.entries(analysis.probabilities).map(([key, val]) => (
                    <li key={key}>
                      <span className="capitalize">{key}</span>: {(val * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Warnings */}
        {analysis.warnings.length > 0 && (
          <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h3 className="font-bold text-yellow-800 mb-2">System Warnings</h3>
            <ul className="list-disc pl-5 text-yellow-700 text-sm">
              {analysis.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-12 pt-6 border-t-2 border-slate-200 text-xs text-slate-500 text-center uppercase tracking-wider font-semibold">
          For Research Use Only. Not for Diagnostic Procedures.
        </div>
      </div>
    </GlassCard>
  );
};
