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
   * List all available flows (names only)
   */
  listFlows: async () => {
    const response = await api.get('/api/flows/list');
    return response.data;
  },

  /**
   * List all available flows with full details in a single request.
   * This is more efficient than calling listFlows + getFlowInfo for each flow.
   * Returns array of flow objects with: name, id, path, description, version, author, tags, recommended_models, dataset_requirements
   */
  listFlowsWithDetails: async () => {
    const response = await api.get('/api/flows/list-with-details');
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

  /**
   * Test model connection with a simple prompt
   * @param {Object} config - Model configuration
   * @param {string} config.model - Model name
   * @param {string} config.api_base - API base URL
   * @param {string} config.api_key - API key
   * @param {string} config.test_prompt - Test prompt to send
   * @returns {Promise<Object>} - Test result with response and latency
   */
  testConnection: async (config) => {
    const response = await api.post('/api/model/test', config);
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

  /**
   * Check for duplicate rows in the loaded dataset
   */
  checkDuplicates: async () => {
    const response = await api.get('/api/dataset/check-duplicates');
    return response.data;
  },

  /**
   * Remove duplicate rows from the loaded dataset
   */
  removeDuplicates: async () => {
    const response = await api.post('/api/dataset/remove-duplicates');
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
// Flow Test API (for Visual Flow Editor)
// ============================================================================

export const flowTestAPI = {
  /**
   * Test flow step-by-step with streaming results
   * @param {Object} config - Test configuration
   * @param {Array} config.blocks - Array of block configurations
   * @param {Object} config.modelConfig - Model configuration (model, api_base, api_key)
   * @param {Object} config.sampleData - Sample input data (column -> value)
   * @param {string} config.workspaceId - Optional workspace ID (if provided, loads blocks from workspace)
   * @param {Function} onEvent - Callback for each SSE event
   * @returns {Promise} - Resolves when stream ends
   */
  testStepByStep: (config, onEvent) => {
    return new Promise((resolve, reject) => {
      // Use fetch with POST for SSE (EventSource only supports GET)
      fetch(`${API_BASE_URL}/api/flow/test-step-by-step`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          blocks: config.blocks,
          model_config_data: config.modelConfig,
          sample_data: config.sampleData,
          workspace_id: config.workspaceId || null,
        }),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          
          const processStream = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                  resolve();
                  break;
                }
                
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete SSE messages
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer
                
                for (const line of lines) {
                  if (line.startsWith('data: ')) {
                    try {
                      const data = JSON.parse(line.slice(6));
                      onEvent(data);
                      
                      // Resolve on completion or error
                      if (data.type === 'test_complete' || data.type === 'test_error') {
                        resolve(data);
                      }
                    } catch (parseError) {
                      console.warn('Failed to parse SSE data:', line, parseError);
                    }
                  }
                }
              }
            } catch (streamError) {
              reject(streamError);
            }
          };
          
          processStream();
        })
        .catch(reject);
    });
  },
};

// ============================================================================
// Workspace Management API (Live Flow Editing)
// ============================================================================

export const workspaceAPI = {
  /**
   * Create a new workspace for flow editing
   * @param {string} sourceFlowName - Optional template flow to clone
   * @returns {Promise<{workspace_id, workspace_path, flow_data, blocks}>}
   */
  create: async (sourceFlowName = null) => {
    const response = await api.post('/api/workspace/create', {
      source_flow_name: sourceFlowName,
    });
    return response.data;
  },

  /**
   * Update the flow.yaml in a workspace
   * @param {string} workspaceId - Workspace ID
   * @param {Object} metadata - Flow metadata
   * @param {Array} blocks - Flow blocks
   */
  updateFlow: async (workspaceId, metadata, blocks) => {
    const response = await api.post(`/api/workspace/${workspaceId}/update-flow`, {
      metadata,
      blocks,
    });
    return response.data;
  },

  /**
   * Create or update a prompt file in the workspace
   * @param {string} workspaceId - Workspace ID
   * @param {string} promptFilename - Filename for the prompt (e.g., "my_prompt.yaml")
   * @param {Object} promptConfig - Prompt configuration (messages array, etc.)
   * @returns {Promise<{status, prompt_filename, full_prompt_path}>}
   */
  updatePrompt: async (workspaceId, promptFilename, promptConfig) => {
    const response = await api.post(`/api/workspace/${workspaceId}/update-prompt`, {
      prompt_filename: promptFilename,
      prompt_config: promptConfig,
    });
    return response.data;
  },

  /**
   * Finalize a workspace (save as permanent flow)
   * @param {string} workspaceId - Workspace ID
   * @param {string} flowName - Final flow name
   * @returns {Promise<{status, flow_name, flow_dir, flow_path}>}
   */
  finalize: async (workspaceId, flowName) => {
    const response = await api.post(`/api/workspace/${workspaceId}/finalize`, {
      flow_name: flowName,
    });
    return response.data;
  },

  /**
   * Delete a workspace (cleanup on cancel)
   * @param {string} workspaceId - Workspace ID
   */
  delete: async (workspaceId) => {
    const response = await api.delete(`/api/workspace/${workspaceId}`);
    return response.data;
  },

  /**
   * Get blocks from a workspace with full prompt paths
   * @param {string} workspaceId - Workspace ID
   * @returns {Promise<{blocks, metadata}>}
   */
  getBlocks: async (workspaceId) => {
    const response = await api.get(`/api/workspace/${workspaceId}/blocks`);
    return response.data;
  },
};

// ============================================================================
// Custom Flows Management API (Browse & Download)
// ============================================================================

export const customFlowsAPI = {
  /**
   * List all custom flows with their files
   * @returns {Promise<{flows: Array}>}
   */
  list: async () => {
    const response = await api.get('/api/custom-flows');
    return response.data;
  },

  /**
   * Get download URL for a specific file
   * @param {string} flowName - Directory name of the custom flow
   * @param {string} filename - Name of the YAML file
   * @returns {string} Download URL
   */
  getFileDownloadUrl: (flowName, filename) => {
    return `${API_BASE_URL}/api/custom-flows/${encodeURIComponent(flowName)}/download/${encodeURIComponent(filename)}`;
  },

  /**
   * Get download URL for all files as ZIP
   * @param {string} flowName - Directory name of the custom flow
   * @returns {string} Download URL for ZIP
   */
  getZipDownloadUrl: (flowName) => {
    return `${API_BASE_URL}/api/custom-flows/${encodeURIComponent(flowName)}/download-all`;
  },

  /**
   * Delete a custom flow
   * @param {string} flowName - Directory name of the custom flow
   */
  delete: async (flowName) => {
    const response = await api.delete(`/api/custom-flows/${encodeURIComponent(flowName)}`);
    return response.data;
  },

  /**
   * Delete all custom flows
   * @returns {Promise<{status: string, deleted_count: number, message: string}>}
   */
  deleteAll: async () => {
    const response = await api.delete('/api/custom-flows');
    return response.data;
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
   * Get all runs for a specific configuration
   */
  listByConfig: async (configId) => {
    const response = await api.get(`/api/runs/config/${configId}`);
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

  /**
   * Preview a generated dataset (first N rows + column names)
   * @param {string} runId - The run ID
   * @param {number} maxRows - Maximum number of rows to return (default 5)
   */
  preview: async (runId, maxRows = 5) => {
    const response = await api.get(`/api/runs/${runId}/preview`, {
      params: { max_rows: maxRows },
    });
    return response.data;
  },

  /**
   * Analyze raw generation logs to extract LLM statistics
   * Sends logs to the server for parsing and stores results in the run record
   */
  analyzeLogs: async (runId, rawLogs) => {
    const response = await api.post(`/api/runs/${runId}/analyze-logs`, {
      raw_logs: rawLogs,
    });
    return response.data;
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
// PDF Preprocessing API
// ============================================================================

export const preprocessingAPI = {
  /**
   * Upload PDF files for preprocessing
   * @param {File[]} files - Array of PDF files to upload
   * @returns {Promise<{job_id: string, files: Array}>}
   */
  uploadPDFs: async (files, existingJobId = null) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    // Build URL with optional existing job ID
    const url = existingJobId 
      ? `/api/preprocessing/upload-pdf?existing_job_id=${encodeURIComponent(existingJobId)}`
      : '/api/preprocessing/upload-pdf';
    
    const response = await api.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Start PDF to Markdown conversion (returns EventSource URL for streaming)
   * @param {string} jobId - The preprocessing job ID
   * @returns {string} The EventSource URL for streaming conversion progress
   */
  getConversionStreamUrl: (jobId) => {
    return `${API_BASE_URL}/api/preprocessing/convert/${jobId}`;
  },

  /**
   * Trigger PDF conversion (POST endpoint, returns streaming response)
   * @param {string} jobId - The preprocessing job ID
   */
  convertPDFs: async (jobId) => {
    // This returns a streaming response, caller should use EventSource or fetch with streaming
    const response = await fetch(`${API_BASE_URL}/api/preprocessing/convert/${jobId}`, {
      method: 'POST',
    });
    return response;
  },

  /**
   * Chunk the converted markdown documents
   * @param {string} jobId - The preprocessing job ID
   * @param {Object} config - Chunking configuration
   * @param {number} config.chunk_size - Max words/tokens per chunk (default: 1000)
   * @param {number} config.overlap - Overlap between chunks (default: 100)
   * @param {string} config.method - Chunking method: 'word' or 'token' (default: 'word')
   */
  chunkDocuments: async (jobId, config = {}) => {
    const payload = {
      chunk_size: config.chunk_size || 1000,
      overlap: config.overlap || 100,
      method: config.method || 'word',
    };
    
    // Add optional per-file parameters
    if (config.selected_files && config.selected_files.length > 0) {
      payload.selected_files = config.selected_files;
    }
    if (config.file_configs) {
      payload.file_configs = config.file_configs;
    }
    
    const response = await api.post(`/api/preprocessing/chunk/${jobId}`, payload);
    return response.data;
  },

  /**
   * Get preprocessing job status
   * @param {string} jobId - The preprocessing job ID
   */
  getStatus: async (jobId) => {
    const response = await api.get(`/api/preprocessing/status/${jobId}`);
    return response.data;
  },

  /**
   * Get chunks from a preprocessing job with pagination
   * @param {string} jobId - The preprocessing job ID
   * @param {number} offset - Pagination offset
   * @param {number} limit - Number of chunks to return
   */
  getChunks: async (jobId, offset = 0, limit = 10) => {
    const response = await api.get(`/api/preprocessing/chunks/${jobId}`, {
      params: { offset, limit },
    });
    return response.data;
  },

  /**
   * Get available ICL templates
   */
  getICLTemplates: async () => {
    const response = await api.get('/api/preprocessing/icl-templates');
    return response.data;
  },

  /**
   * Create a dataset from preprocessed documents
   * @param {string} jobId - The preprocessing job ID
   * @param {Object} options - Dataset creation options
   * @param {Object} options.chunk_config - Chunking configuration used
   * @param {Object} options.additional_columns - Additional columns to add
   * @param {Object} options.icl_template - ICL template to use
   * @param {string} options.domain - Document domain
   * @param {string} options.document_outline - Document outline/description
   * @param {string} options.content_column_name - Name of content column ('text' or 'document')
   * @param {boolean} options.include_domain - Whether to include domain column
   * @param {boolean} options.include_document_outline - Whether to include document_outline column
   */
  createDataset: async (jobId, options = {}) => {
    const response = await api.post(`/api/preprocessing/create-dataset/${jobId}`, {
      job_id: jobId,
      chunk_config: options.chunk_config || { chunk_size: 1000, overlap: 100, method: 'word' },
      additional_columns: options.additional_columns || {},
      icl_template: options.icl_template || null,
      domain: options.domain || '',
      document_outline: options.document_outline || '',
      dataset_name: options.dataset_name || null,  // Custom name for the dataset
      // Flow-aware column configuration
      content_column_name: options.content_column_name || 'document',
      include_domain: options.include_domain !== undefined ? options.include_domain : true,
      include_document_outline: options.include_document_outline !== undefined ? options.include_document_outline : true,
    });
    return response.data;
  },

  /**
   * List all preprocessing jobs with converted files
   * @returns {Promise<{jobs: Array, total: number}>}
   */
  listJobs: async () => {
    const response = await api.get('/api/preprocessing/jobs');
    return response.data;
  },

  /**
   * List all preprocessed datasets (final JSONL files)
   * @returns {Promise<{datasets: Array, total: number}>}
   */
  listDatasets: async () => {
    const response = await api.get('/api/preprocessing/datasets');
    return response.data;
  },

  /**
   * Delete a preprocessed dataset
   * @param {string} jobId - The preprocessing job ID
   */
  deleteDataset: async (jobId) => {
    const response = await api.delete(`/api/preprocessing/datasets/${jobId}`);
    return response.data;
  },

  /**
   * Get download URL for a preprocessed dataset
   * @param {string} jobId - The preprocessing job ID
   * @returns {string} The download URL
   */
  getDatasetDownloadUrl: (jobId) => {
    return `${API_BASE_URL}/api/preprocessing/datasets/${jobId}/download`;
  },

  /**
   * Get download URL for a converted markdown file
   * @param {string} jobId - The preprocessing job ID
   * @param {string} filename - The markdown filename
   * @returns {string} The download URL
   */
  getDownloadUrl: (jobId, filename) => {
    return `${API_BASE_URL}/api/preprocessing/download/${jobId}/${encodeURIComponent(filename)}`;
  },

  /**
   * Clean up a preprocessing job
   * @param {string} jobId - The preprocessing job ID
   */
  cleanup: async (jobId) => {
    const response = await api.delete(`/api/preprocessing/${jobId}`);
    return response.data;
  },

  /**
   * Get URL to view an uploaded PDF file
   * @param {string} jobId - The preprocessing job ID
   * @param {string} filename - The PDF filename
   * @returns {string} The PDF URL
   */
  getPdfUrl: (jobId, filename) => {
    return `${API_BASE_URL}/api/preprocessing/pdf/${jobId}/${encodeURIComponent(filename)}`;
  },

  /**
   * Get markdown content as text
   * @param {string} jobId - The preprocessing job ID
   * @param {string} filename - The markdown filename
   * @returns {Promise<{content: string, filename: string}>}
   */
  getMarkdownContent: async (jobId, filename) => {
    const response = await api.get(`/api/preprocessing/markdown-content/${jobId}/${encodeURIComponent(filename)}`);
    return response.data;
  },
};

// ============================================================================
// Health Check
// ============================================================================

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Export default api instance for custom requests
export default api;
