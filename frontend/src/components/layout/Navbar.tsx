import React from 'react';
import { NavLink } from 'react-router-dom';
import { Dna, Menu, X, Info, Home, FileSearch, BarChart2 } from 'lucide-react';
import { useModelStatus } from '../../hooks/useModelStatus';

export const Navbar: React.FC = () => {
  const { isConnected, modelStatus } = useModelStatus();
  const [isOpen, setIsOpen] = React.useState(false);

  const modelsLoaded = modelStatus?.unet && modelStatus?.resnet50;

  const links = [
    { to: '/', label: 'Home', icon: Home },
    { to: '/analysis', label: 'Analysis', icon: FileSearch },
    { to: '/model-info', label: 'Model Info', icon: Info },
    { to: '/evaluation', label: 'Evaluation', icon: BarChart2 },
  ];

  return (
    <nav className="fixed w-full z-50 glass border-b border-white/10 top-0">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <NavLink to="/" className="flex items-center gap-2">
              <Dna className="w-8 h-8 text-blue-500" />
              <span className="font-bold text-xl tracking-tight text-white">
                Pancreas<span className="text-gradient">AI</span>
              </span>
            </NavLink>
          </div>
          
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-8">
              {links.map((link) => {
                const Icon = link.icon;
                return (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    className={({ isActive }) =>
                      `flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? 'text-white bg-white/10'
                          : 'text-slate-300 hover:text-white hover:bg-white/5'
                      }`
                    }
                  >
                    <Icon className="w-4 h-4" />
                    {link.label}
                  </NavLink>
                );
              })}
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">Models:</span>
              {modelsLoaded ? (
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-full text-xs">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Loaded
                </span>
              ) : (
                <span className="flex items-center gap-1 text-amber-400 bg-amber-400/10 px-2 py-1 rounded-full text-xs">
                  <span className="w-2 h-2 rounded-full bg-amber-500"></span> Pending
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">API:</span>
              <div className="flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]'}`}></span>
                <span className={isConnected ? 'text-emerald-400' : 'text-red-400'}>
                  {isConnected ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
          </div>

          <div className="-mr-2 flex md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-white hover:bg-white/10"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {isOpen && (
        <div className="md:hidden glass border-t border-white/10">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {links.map((link) => {
              const Icon = link.icon;
              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  onClick={() => setIsOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-3 py-2 rounded-md text-base font-medium ${
                      isActive
                        ? 'text-white bg-white/10'
                        : 'text-slate-300 hover:text-white hover:bg-white/5'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  {link.label}
                </NavLink>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
};
