import React, { useState } from 'react';
import { Maximize2, Download, Image as ImageIcon, X, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

interface ImageViewerProps {
  src: string | null;
  title: string;
  subtitle?: string;
  className?: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({ src, title, subtitle, className = '' }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handleZoomIn = () => setScale(prev => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setScale(prev => Math.max(prev - 0.25, 0.5));
  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) handleZoomIn();
    else handleZoomOut();
  };

  const downloadImage = () => {
    if (!src) return;
    const link = document.createElement('a');
    link.href = src.startsWith('data:') ? src : `data:image/png;base64,${src}`;
    link.download = `${title.toLowerCase().replace(/\s+/g, '_')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const imageSrc = src ? (src.startsWith('data:') ? src : `data:image/png;base64,${src}`) : null;

  const ImageContent = () => (
    <div 
      className="relative w-full h-full min-h-[250px] bg-black/40 rounded-lg overflow-hidden flex items-center justify-center border border-white/5"
      onWheel={handleWheel}
    >
      {!imageSrc ? (
        <div className="flex flex-col items-center text-slate-500">
          <ImageIcon className="w-12 h-12 mb-2 opacity-50" />
          <span className="text-sm">Image unavailable</span>
        </div>
      ) : (
        <div 
          className={`w-full h-full flex items-center justify-center ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <img 
            src={imageSrc} 
            alt={title}
            className="max-w-full max-h-full object-contain pointer-events-none transition-transform duration-75"
            style={{ transform: `translate(${position.x}px, ${position.y}px) scale(${scale})` }}
          />
        </div>
      )}

      {/* Controls Overlay */}
      {imageSrc && (
        <div className="absolute bottom-4 right-4 flex items-center gap-2 bg-slate-900/80 backdrop-blur p-2 rounded-lg border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity">
          <button onClick={handleZoomOut} className="p-1.5 hover:bg-white/10 rounded text-slate-300 hover:text-white"><ZoomOut className="w-4 h-4" /></button>
          <button onClick={handleReset} className="p-1.5 hover:bg-white/10 rounded text-slate-300 hover:text-white"><RefreshCw className="w-4 h-4" /></button>
          <button onClick={handleZoomIn} className="p-1.5 hover:bg-white/10 rounded text-slate-300 hover:text-white"><ZoomIn className="w-4 h-4" /></button>
          <div className="w-px h-4 bg-white/20 mx-1"></div>
          <button onClick={downloadImage} className="p-1.5 hover:bg-white/10 rounded text-slate-300 hover:text-white"><Download className="w-4 h-4" /></button>
          {!isFullscreen && (
            <button onClick={() => setIsFullscreen(true)} className="p-1.5 hover:bg-white/10 rounded text-slate-300 hover:text-white"><Maximize2 className="w-4 h-4" /></button>
          )}
        </div>
      )}
    </div>
  );

  return (
    <>
      <div className={`flex flex-col h-full group ${className}`}>
        <div className="mb-3">
          <h4 className="font-medium text-slate-200">{title}</h4>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
        <div className="flex-1 relative">
          <ImageContent />
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-[100] bg-black/95 flex flex-col backdrop-blur-sm">
          <div className="flex items-center justify-between p-4 border-b border-white/10 bg-slate-900/50">
            <div>
              <h3 className="text-xl font-medium text-white">{title}</h3>
              {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
            </div>
            <button 
              onClick={() => { setIsFullscreen(false); handleReset(); }}
              className="p-2 hover:bg-white/10 rounded-full text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <div className="flex-1 p-4 md:p-8 flex items-center justify-center overflow-hidden relative group">
            <ImageContent />
          </div>
        </div>
      )}
    </>
  );
};
