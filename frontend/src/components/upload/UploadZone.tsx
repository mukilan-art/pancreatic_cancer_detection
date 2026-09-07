import React, { useCallback, useRef, useState } from 'react';
import { UploadCloud, File, X, AlertCircle } from 'lucide-react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onFileSelect, disabled = false }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const validateAndProcessFile = async (file: File) => {
    setError(null);
    
    // Check file type
    const validTypes = ['image/jpeg', 'image/png', 'application/dicom'];
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm')) {
      setError('Invalid file type. Please upload a PNG, JPG, or DICOM file.');
      return;
    }

    // Check file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('File is too large. Maximum size is 50MB.');
      return;
    }

    if (file.type.startsWith('image/')) {
      try {
        const bitmap = await createImageBitmap(file);
        bitmap.close();
      } catch {
        setError('The selected file is not a readable PNG or JPG image.');
        return;
      }
    }

    setSelectedFile(file);
    if (file.type.startsWith('image/')) {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
    } else {
      setPreviewUrl(null); // No preview for DICOM yet
    }
    
    onFileSelect(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      void validateAndProcessFile(e.dataTransfer.files[0]);
    }
  }, [disabled, onFileSelect]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      void validateAndProcessFile(e.target.files[0]);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (selectedFile) {
    return (
      <div className="glass rounded-xl p-6 relative border-emerald-500/30 bg-emerald-500/5">
        <button 
          onClick={clearFile}
          className="absolute top-4 right-4 p-2 bg-slate-800/50 rounded-full hover:bg-slate-700 transition-colors"
        >
          <X className="w-5 h-5 text-slate-300" />
        </button>
        
        <div className="flex flex-col items-center justify-center space-y-4">
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="max-h-48 rounded-lg object-contain bg-black/50" />
          ) : (
            <div className="w-24 h-24 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <File className="w-12 h-12 text-blue-400" />
            </div>
          )}
          <div className="text-center">
            <h4 className="text-lg font-medium text-white">{selectedFile.name}</h4>
            <p className="text-sm text-slate-400">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
          <div className="flex items-center gap-2 text-emerald-400 bg-emerald-400/10 px-3 py-1.5 rounded-full text-sm font-medium">
            <CheckCircle2 className="w-4 h-4" /> Ready for analysis
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.dcm"
        onChange={handleChange}
        className="hidden"
        disabled={disabled}
      />
      
      <div 
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`
          relative border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all
          ${disabled ? 'opacity-50 cursor-not-allowed border-slate-700 bg-slate-800/20' : ''}
          ${isDragging ? 'border-blue-400 bg-blue-500/10 scale-[1.02]' : 'border-slate-600 hover:border-blue-500/50 hover:bg-white/5'}
          ${error ? 'border-red-500/50 bg-red-500/5' : ''}
        `}
      >
        <div className={`p-4 rounded-full mb-4 ${isDragging ? 'bg-blue-500/20 text-blue-400 animate-pulse' : 'bg-slate-800 text-slate-400'}`}>
          <UploadCloud className="w-10 h-10" />
        </div>
        
        <h3 className="text-lg font-semibold text-white mb-2">
          {isDragging ? 'Drop file here' : 'Drag & Drop your CT Scan'}
        </h3>
        <p className="text-sm text-slate-400 mb-6 max-w-xs mx-auto">
          Upload an abdominal CT scan image to begin the explainable AI analysis pipeline.
        </p>
        
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="px-2 py-1 bg-slate-800 rounded">PNG</span>
          <span className="px-2 py-1 bg-slate-800 rounded">JPG</span>
          <span className="px-2 py-1 bg-slate-800 rounded">DICOM</span>
          <span className="ml-2">Max 50MB</span>
        </div>

        {error && (
          <div className="mt-6 flex items-center gap-2 text-red-400 bg-red-400/10 px-4 py-2 rounded-lg text-sm">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

function CheckCircle2(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}
