import React from 'react';
import { ImageViewer } from './ImageViewer';
import { GlassCard } from '../ui/GlassCard';
import { Layers } from 'lucide-react';

interface ImageGalleryProps {
  images: Array<{
    src: string | null;
    title: string;
    subtitle?: string;
  }>;
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({ images }) => {
  if (!images || images.length === 0) return null;

  return (
    <GlassCard title="Processing Pipeline Images" icon={<Layers className="w-5 h-5" />}>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mt-4">
        {images.map((img, idx) => (
          <ImageViewer 
            key={idx} 
            src={img.src} 
            title={img.title} 
            subtitle={img.subtitle} 
          />
        ))}
      </div>
    </GlassCard>
  );
};
