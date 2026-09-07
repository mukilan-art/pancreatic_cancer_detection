import React, { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  hoverEffect?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({ 
  children, 
  className = '', 
  title, 
  subtitle, 
  icon,
  hoverEffect = false 
}) => {
  return (
    <div className={`glass rounded-xl overflow-hidden ${hoverEffect ? 'glass-hover' : ''} ${className}`}>
      {(title || icon) && (
        <div className="p-4 border-b border-white/5 flex items-center gap-3 bg-white/5">
          {icon && <div className="text-blue-400">{icon}</div>}
          <div>
            {title && <h3 className="font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
          </div>
        </div>
      )}
      <div className="p-4">
        {children}
      </div>
    </div>
  );
};
