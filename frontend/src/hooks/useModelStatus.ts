import { useState, useEffect } from 'react';
import { ModelStatus } from '../types';
import api from '../services/api';

export const useModelStatus = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const checkStatus = async () => {
    try {
      const status = await api.getModelStatus();
      setModelStatus(status);
      setIsConnected(true);
      setLastChecked(new Date());
    } catch (error) {
      setIsConnected(false);
      setModelStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return { modelStatus, isConnected, isLoading, lastChecked, checkStatus };
};
