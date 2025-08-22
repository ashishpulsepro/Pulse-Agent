import React from 'react';
import { useState, useEffect } from 'react';
import apiService from '../services/api';

const StatusIndicator = () => {
  const [status, setStatus] = useState('checking');
  const [details, setDetails] = useState(null);

  useEffect(() => {
    checkSystemHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      const [healthResponse, chatHealthResponse] = await Promise.allSettled([
        apiService.checkHealth(),
        apiService.checkChatHealth()
      ]);

      const isHealthy = healthResponse.status === 'fulfilled';
      const isChatHealthy = chatHealthResponse.status === 'fulfilled' && 
                           chatHealthResponse.value?.success;

      if (isHealthy && isChatHealthy) {
        setStatus('healthy');
        setDetails({
          api: 'Connected',
          chat: 'Ready',
          ollama: chatHealthResponse.value?.data?.model || 'Unknown'
        });
      } else {
        setStatus('degraded');
        setDetails({
          api: isHealthy ? 'Connected' : 'Error',
          chat: isChatHealthy ? 'Ready' : 'Error',
          ollama: 'Check Required'
        });
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setStatus('error');
      setDetails({
        api: 'Disconnected',
        chat: 'Unavailable',
        ollama: 'Offline'
      });
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-50 border-green-200';
      case 'degraded': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'error': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'healthy':
        return (
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'degraded':
        return (
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      default:
        return (
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"></div>
        );
    }
  };

  return (
    <div className={`inline-flex items-center px-2.5 py-1.5 rounded-md text-xs font-medium border ${getStatusColor()}`}>
      {getStatusIcon()}
      <span className="ml-1.5">
        {status === 'checking' ? 'Checking...' :
         status === 'healthy' ? 'All Systems Online' :
         status === 'degraded' ? 'Partial Service' :
         'Service Unavailable'}
      </span>
      
      {details && status !== 'checking' && (
        <div className="ml-2 text-xs opacity-75">
          API: {details.api} | Chat: {details.chat}
        </div>
      )}
    </div>
  );
};

export default StatusIndicator;
