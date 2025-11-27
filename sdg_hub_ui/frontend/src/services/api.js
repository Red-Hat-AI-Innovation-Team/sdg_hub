/**
 * API Service for SDG Hub
 * 
 * Provides methods to interact with the SDG Hub backend API.
 */

import axios from 'axios';

/**
 * Determine the API base URL dynamically.
 * This allows running multiple instances on different ports:
 * - Frontend 3000 -> Backend 8000 (default)
 * - Frontend 3001 -> Backend 8001
 * - Frontend 3002 -> Backend 8002
 * - etc.
 */
const getApiBaseUrl = () => {
  // First check for explicit environment variable (for production builds)
  if (process.env.REACT_APP_API_URL) {
    console.log(`🔗 Using env API URL: ${process.env.REACT_APP_API_URL}`);
    return process.env.REACT_APP_API_URL;
  }
  
  // Dynamic port mapping for development/demo instances
  const frontendPort = window.location.port;
  const hostname = window.location.hostname;
  
  if (frontendPort && frontendPort !== '3000') {
    // Map frontend port to backend port (frontend 300X -> backend 800X)
    const backendPort = frontendPort.replace('300', '800');
    const apiUrl = `http://${hostname}:${backendPort}`;
    console.log(`🔗 Dynamic API mapping: Frontend :${frontendPort} -> Backend :${backendPort}`);
    console.log(`🔗 Full API URL: ${apiUrl}`);
    return apiUrl;
  }
  
  // Default
  const defaultUrl = `http://${hostname || 'localhost'}:8000`;
  console.log(`🔗 Using default API URL: ${defaultUrl}`);
  return defaultUrl;
};

export const API_BASE_URL = getApiBaseUrl();
console.log(`📡 API_BASE_URL initialized to: ${API_BASE_URL}`);

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============================================================================
// Flow Discovery API
// ============================================================================

export const flowAPI = {
  /**
   * List all available flows
   */
  listFlows: async () => {
    const response = await api.get('/api/flows/list');
    return response.data;
  },

  /**
   * Search flows by tag or name
   */
  searchFlows: async (tag = null, nameFilter = null) => {
    const response = await api.post('/api/flows/search', {
      tag,
      name_filter: nameFilter,
    });
    return response.data;
  },

  /**
   * Get detailed information about a specific flow
   */
  getFlowInfo: async (flowName) => {
    const response = await api.get(`/api/flows/${encodeURIComponent(flowName)}/info`);
    return response.data;
  },

  /**
   * Select a flow for configuration
   */
  selectFlow: async (flowName) => {
    const response = await api.post(`/api/flows/${encodeURIComponent(flowName)}/select`);
    return response.data;
  },

  /**
   * Save a custom flow to the server
   */
  saveCustomFlow: async (flowData) => {
    const response = await api.post('/api/flows/save-custom', flowData);
    return response.data;
  },
};

// ============================================================================
// Model Configuration API
// ============================================================================

export const modelAPI = {
  /**
   * Get model recommendations for the selected flow
   */
  getRecommendations: async () => {
    const response = await api.get('/api/model/recommendations');
    return response.data;
  },

  /**
   * Configure model settings
   */
  configure: async (config) => {
    const response = await api.post('/api/model/configure', config);
    return response.data;
  },
};

// ============================================================================
// Dataset Management API
// ============================================================================

export const datasetAPI = {
  /**
   * Upload a dataset file
   */
  uploadFile: async (file) => {
    console.log('uploadFile called with:', file);
    console.log('File type:', file?.constructor?.name);
    console.log('File name:', file?.name);
    console.log('File size:', file?.size);
    
    if (!(file instanceof File)) {
      console.error('ERROR: file is not a File instance!');
      throw new Error('Invalid file object');
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('FormData entries:', [...formData.entries()]);
    
    // Use native fetch to avoid any axios header issues
    const response = await fetch(`${API_BASE_URL}/api/dataset/upload`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type - browser will set it with boundary
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload error response:', errorText);
      throw new Error(errorText || 'Upload failed');
    }
    
    return response.json();
  },

  /**
   * Load dataset from file
   */
  loadDataset: async (config) => {
    const response = await api.post('/api/dataset/load', config);
    return response.data;
  },

  /**
   * Get the required dataset schema
   */
  getSchema: async () => {
    const response = await api.get('/api/dataset/schema');
    return response.data;
  },

  /**
   * Get a preview of the loaded dataset
   */
  getPreview: async () => {
    const response = await api.get('/api/dataset/preview');
    return response.data;
  },
};

// ============================================================================
// Flow Execution API
// ============================================================================

export const executionAPI = {
  /**
   * Perform a dry run
   */
  dryRun: async (config) => {
    const response = await api.post('/api/flow/dry-run', config);
    return response.data;
  },

  /**
   * Cancel current generation (optionally scoped to a configuration)
   */
  cancel: async (configId) => {
    const url = configId 
      ? `/api/flow/cancel-generation?config_id=${encodeURIComponent(configId)}`
      : '/api/flow/cancel-generation';
    const response = await api.post(url);
    return response.data;
  },

  /**
   * Check generation status (to detect running generations after page refresh)
   */
  getGenerationStatus: async (configId = null) => {
    const url = configId 
      ? `/api/flow/generation-status?config_id=${encodeURIComponent(configId)}`
      : '/api/flow/generation-status';
    const response = await api.get(url);
    return response.data;
  },

  /**
   * Get the URL for reconnecting to an existing generation stream
   */
  getReconnectStreamUrl: (configId) => {
    return `${API_BASE_URL}/api/flow/reconnect-stream?config_id=${encodeURIComponent(configId)}`;
  },
};

// ============================================================================
// Checkpoint Management API
// ============================================================================

export const checkpointAPI = {
  /**
   * Get checkpoint information for a configuration
   */
  getCheckpointInfo: async (configId) => {
    const response = await api.get(`/api/flow/checkpoints/${encodeURIComponent(configId)}`);
    return response.data;
  },

  /**
   * Clear checkpoints for a configuration
   */
  clearCheckpoints: async (configId) => {
    const response = await api.delete(`/api/flow/checkpoints/${encodeURIComponent(configId)}`);
    return response.data;
  },
};

// ============================================================================
// Configuration Management API
// ============================================================================

export const configAPI = {
  /**
   * Get current configuration state
   */
  getCurrent: async () => {
    const response = await api.get('/api/config/current');
    return response.data;
  },

  /**
   * Reset configuration
   */
  reset: async () => {
    const response = await api.post('/api/config/reset');
    return response.data;
  },

  /**
   * Import configuration from file
   */
  importConfig: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    // Don't set Content-Type header - let axios set it with the boundary
    const response = await api.post('/api/config/import', formData);
    return response.data;
  },
};

// ============================================================================
// Saved Configurations API
// ============================================================================

export const savedConfigAPI = {
  /**
   * List all saved configurations
   */
  list: async () => {
    const response = await api.get('/api/configurations/list');
    return response.data;
  },

  /**
   * Get a specific configuration
   */
  get: async (configId) => {
    const response = await api.get(`/api/configurations/${configId}`);
    return response.data;
  },

  /**
   * Save a configuration
   */
  save: async (configData) => {
    const response = await api.post('/api/configurations/save', configData);
    return response.data;
  },

  /**
   * Delete a configuration
   */
  delete: async (configId) => {
    const response = await api.delete(`/api/configurations/${configId}`);
    return response.data;
  },

  /**
   * Load a saved configuration
   */
  load: async (configId) => {
    const response = await api.post(`/api/configurations/${configId}/load`);
    return response.data;
  },
};

// ============================================================================
// Block Registry API
// ============================================================================

export const blockAPI = {
  /**
   * List all available blocks
   */
  listBlocks: async () => {
    const response = await api.get('/api/blocks/list');
    return response.data;
  },
};

// ============================================================================
// Flow Runs API
// ============================================================================

export const runsAPI = {
  /**
   * Get all flow runs history
   */
  list: async () => {
    const response = await api.get('/api/runs/list');
    return response.data;
  },

  /**
   * Get a specific run by ID
   */
  get: async (runId) => {
    const response = await api.get(`/api/runs/${runId}`);
    return response.data;
  },

  /**
   * Create a new run record
   */
  create: async (runData) => {
    const response = await api.post('/api/runs/create', runData);
    return response.data;
  },

  /**
   * Update a run record
   */
  update: async (runId, updates) => {
    const response = await api.put(`/api/runs/${runId}/update`, updates);
    return response.data;
  },

  /**
   * Delete a run record
   */
  delete: async (runId) => {
    const response = await api.delete(`/api/runs/${runId}`);
    return response.data;
  },

  /**
   * Download the output dataset from a run
   */
  download: async (runId) => {
    const response = await api.get(`/api/runs/${runId}/download`, {
      responseType: 'blob', // Important for file downloads
    });
    
    // Create a blob URL and trigger download
    const blob = new Blob([response.data], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `run_${runId}_output.jsonl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return { status: 'success', message: 'Dataset downloaded successfully' };
  },
};

// ============================================================================
// Prompt Management API
// ============================================================================

export const promptAPI = {
  /**
   * Save a prompt template
   */
  savePrompt: async (promptData) => {
    const response = await api.post('/api/prompts/save', promptData);
    return response.data;
  },

  /**
   * Load a prompt template
   */
  loadPrompt: async (promptPath) => {
    const response = await api.get(`/api/prompts/load`, {
      params: { prompt_path: promptPath }
    });
    return response.data;
  },
};

// ============================================================================
// Evaluation API (MDM Integration) - REMOVED FOR CURRENT RELEASE
// Note: This feature is preserved in the UI_BackUp folder for future releases
// ============================================================================

// ============================================================================
// Health Check
// ============================================================================

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Export default api instance for custom requests
export default api;
