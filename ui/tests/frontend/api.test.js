// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for API service module.
 */

import axios from 'axios';
import {
  flowAPI,
  modelAPI,
  datasetAPI,
  executionAPI,
  checkpointAPI,
  configAPI,
  savedConfigAPI,
  blockAPI,
  runsAPI,
  promptAPI,
  healthCheck,
  API_BASE_URL,
} from '../../frontend/src/services/api';

// Mock axios
jest.mock('axios');

const mockAxios = axios;

describe('API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Setup axios mock instance
    mockAxios.create.mockReturnValue(mockAxios);
    mockAxios.interceptors = {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    };
  });

  describe('API_BASE_URL', () => {
    it('should have default base URL', () => {
      expect(API_BASE_URL).toBeDefined();
      expect(typeof API_BASE_URL).toBe('string');
    });
  });

  describe('healthCheck', () => {
    it('should call health endpoint', async () => {
      mockAxios.get.mockResolvedValueOnce({
        data: { status: 'healthy', service: 'sdg_hub_api' },
      });

      const result = await healthCheck();
      
      expect(result).toEqual({ status: 'healthy', service: 'sdg_hub_api' });
    });
  });

  describe('flowAPI', () => {
    describe('listFlows', () => {
      it('should list all flows', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: ['Flow 1', 'Flow 2'],
        });

        const result = await flowAPI.listFlows();
        
        expect(result).toEqual(['Flow 1', 'Flow 2']);
      });
    });

    describe('searchFlows', () => {
      it('should search flows by tag', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: ['Matching Flow'],
        });

        const result = await flowAPI.searchFlows('test-tag');
        
        expect(result).toEqual(['Matching Flow']);
      });

      it('should search flows by name filter', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: ['Filtered Flow'],
        });

        const result = await flowAPI.searchFlows(null, 'filter');
        
        expect(result).toEqual(['Filtered Flow']);
      });
    });

    describe('getFlowInfo', () => {
      it('should get flow info', async () => {
        const flowInfo = {
          name: 'Test Flow',
          id: 'test-id',
          description: 'A test flow',
        };
        mockAxios.get.mockResolvedValueOnce({ data: flowInfo });

        const result = await flowAPI.getFlowInfo('Test Flow');
        
        expect(result).toEqual(flowInfo);
      });
    });

    describe('selectFlow', () => {
      it('should select a flow', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', message: 'Flow selected' },
        });

        const result = await flowAPI.selectFlow('Test Flow');
        
        expect(result.status).toBe('success');
      });
    });

    describe('saveCustomFlow', () => {
      it('should save a custom flow', async () => {
        const flowData = {
          metadata: { name: 'My Custom Flow' },
          blocks: [],
        };
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', flow_path: '/path/to/flow.yaml' },
        });

        const result = await flowAPI.saveCustomFlow(flowData);
        
        expect(result.status).toBe('success');
        expect(result.flow_path).toBeDefined();
      });
    });
  });

  describe('modelAPI', () => {
    describe('getRecommendations', () => {
      it('should get model recommendations', async () => {
        const recommendations = {
          default_model: 'test-model',
          recommendations: { default: 'test-model', compatible: [] },
        };
        mockAxios.get.mockResolvedValueOnce({ data: recommendations });

        const result = await modelAPI.getRecommendations();
        
        expect(result.default_model).toBe('test-model');
      });
    });

    describe('configure', () => {
      it('should configure model', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', message: 'Model configured' },
        });

        const result = await modelAPI.configure({
          model: 'test-model',
          api_base: 'http://localhost:8000/v1',
          api_key: 'EMPTY',
        });
        
        expect(result.status).toBe('success');
      });
    });
  });

  describe('datasetAPI', () => {
    describe('uploadFile', () => {
      it('should upload a file', async () => {
        // uploadFile uses native fetch, not axios
        global.fetch = jest.fn().mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'success', file_path: 'uploads/test.jsonl' }),
        });

        const file = new File(['{"test": "data"}'], 'test.jsonl', {
          type: 'application/json',
        });
        const result = await datasetAPI.uploadFile(file);
        
        expect(result.status).toBe('success');
        expect(result.file_path).toBeDefined();
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/dataset/upload'),
          expect.objectContaining({ method: 'POST' })
        );
      });
    });

    describe('loadDataset', () => {
      it('should load dataset', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: {
            status: 'success',
            num_samples: 100,
            columns: ['input', 'output'],
          },
        });

        const result = await datasetAPI.loadDataset({
          data_files: 'test.jsonl',
          file_format: 'auto',
        });
        
        expect(result.status).toBe('success');
        expect(result.num_samples).toBe(100);
      });
    });

    describe('getSchema', () => {
      it('should get dataset schema', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { columns: ['input'], requirements: null },
        });

        const result = await datasetAPI.getSchema();
        
        expect(result.columns).toContain('input');
      });
    });

    describe('getPreview', () => {
      it('should get dataset preview', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: {
            num_samples: 10,
            columns: ['input'],
            preview: { input: ['test'] },
          },
        });

        const result = await datasetAPI.getPreview();
        
        expect(result.num_samples).toBe(10);
        expect(result.preview).toBeDefined();
      });
    });
  });

  describe('executionAPI', () => {
    describe('dryRun', () => {
      it('should perform dry run', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { flow_name: 'Test Flow', status: 'success' },
        });

        const result = await executionAPI.dryRun({
          sample_size: 2,
          enable_time_estimation: true,
        });
        
        expect(result.status).toBe('success');
      });
    });

    describe('cancel', () => {
      it('should cancel generation', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', message: 'Cancelled' },
        });

        const result = await executionAPI.cancel();
        
        expect(result).toBeDefined();
        expect(result.status).toBe('success');
      });

      it('should cancel generation for specific config', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', message: 'Cancelled' },
        });

        const result = await executionAPI.cancel('config-123');
        
        expect(result.status).toBe('success');
      });
    });

    describe('getGenerationStatus', () => {
      it('should get generation status', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { running_generations: [] },
        });

        const result = await executionAPI.getGenerationStatus();
        
        expect(result.running_generations).toEqual([]);
      });
    });

    describe('getReconnectStreamUrl', () => {
      it('should return reconnect stream URL', () => {
        const url = executionAPI.getReconnectStreamUrl('config-123');
        
        expect(url).toContain('reconnect-stream');
        expect(url).toContain('config_id=config-123');
      });
    });
  });

  describe('checkpointAPI', () => {
    describe('getCheckpointInfo', () => {
      it('should get checkpoint info', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { has_checkpoints: false, checkpoint_count: 0 },
        });

        const result = await checkpointAPI.getCheckpointInfo('config-123');
        
        expect(result.has_checkpoints).toBe(false);
      });
    });

    describe('clearCheckpoints', () => {
      it('should clear checkpoints', async () => {
        mockAxios.delete.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await checkpointAPI.clearCheckpoints('config-123');
        
        expect(result.status).toBe('success');
      });
    });
  });

  describe('configAPI', () => {
    describe('getCurrent', () => {
      it('should get current config', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { flow: null, model_config: {}, dataset: null },
        });

        const result = await configAPI.getCurrent();
        
        expect(result).toBeDefined();
      });
    });

    describe('reset', () => {
      it('should reset config', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await configAPI.reset();
        
        expect(result.status).toBe('success');
      });
    });

    describe('importConfig', () => {
      it('should import config file', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', configuration: {} },
        });

        const file = new File(['{}'], 'config.json', {
          type: 'application/json',
        });
        const result = await configAPI.importConfig(file);
        
        expect(result.status).toBe('success');
      });
    });
  });

  describe('savedConfigAPI', () => {
    describe('list', () => {
      it('should list saved configurations', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: [{ id: 'config-1', flow_name: 'Test Flow' }],
        });

        const result = await savedConfigAPI.list();
        
        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('get', () => {
      it('should get specific configuration', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { id: 'config-1', flow_name: 'Test Flow' },
        });

        const result = await savedConfigAPI.get('config-1');
        
        expect(result.id).toBe('config-1');
      });
    });

    describe('save', () => {
      it('should save configuration', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success', configuration: { id: 'new-config' } },
        });

        const result = await savedConfigAPI.save({
          flow_name: 'Test Flow',
          model_configuration: {},
        });
        
        expect(result.status).toBe('success');
      });
    });

    describe('delete', () => {
      it('should delete configuration', async () => {
        mockAxios.delete.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await savedConfigAPI.delete('config-1');
        
        expect(result.status).toBe('success');
      });
    });

    describe('load', () => {
      it('should load configuration', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await savedConfigAPI.load('config-1');
        
        expect(result.status).toBe('success');
      });
    });
  });

  describe('blockAPI', () => {
    describe('listBlocks', () => {
      it('should list available blocks', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { blocks: ['LLMChatBlock', 'TextConcatBlock'] },
        });

        const result = await blockAPI.listBlocks();
        
        expect(result.blocks).toBeDefined();
      });
    });
  });

  describe('runsAPI', () => {
    describe('list', () => {
      it('should list runs', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: [{ run_id: 'run-1', status: 'completed' }],
        });

        const result = await runsAPI.list();
        
        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('get', () => {
      it('should get specific run', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { run_id: 'run-1', status: 'completed' },
        });

        const result = await runsAPI.get('run-1');
        
        expect(result.run_id).toBe('run-1');
      });
    });

    describe('create', () => {
      it('should create run record', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await runsAPI.create({
          run_id: 'run-new',
          flow_name: 'Test Flow',
          status: 'running',
        });
        
        expect(result.status).toBe('success');
      });
    });

    describe('update', () => {
      it('should update run record', async () => {
        mockAxios.put.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await runsAPI.update('run-1', {
          status: 'completed',
        });
        
        expect(result.status).toBe('success');
      });
    });

    describe('delete', () => {
      it('should delete run record', async () => {
        mockAxios.delete.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await runsAPI.delete('run-1');
        
        expect(result.status).toBe('success');
      });
    });
  });

  describe('promptAPI', () => {
    describe('savePrompt', () => {
      it('should save prompt', async () => {
        mockAxios.post.mockResolvedValueOnce({
          data: { status: 'success' },
        });

        const result = await promptAPI.savePrompt({
          prompt_path: '/path/to/prompt.yaml',
          content: 'Test prompt content',
        });
        
        expect(result.status).toBe('success');
      });
    });

    describe('loadPrompt', () => {
      it('should load prompt', async () => {
        mockAxios.get.mockResolvedValueOnce({
          data: { content: 'Test prompt content' },
        });

        const result = await promptAPI.loadPrompt('/path/to/prompt.yaml');
        
        expect(result.content).toBeDefined();
      });
    });
  });
});

describe('API Error Handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockAxios.create.mockReturnValue(mockAxios);
  });

  it('should handle network errors', async () => {
    mockAxios.get.mockRejectedValueOnce(new Error('Network Error'));

    await expect(healthCheck()).rejects.toThrow('Network Error');
  });

  it('should handle 404 errors', async () => {
    mockAxios.get.mockRejectedValueOnce({
      response: {
        status: 404,
        data: { detail: 'Not found' },
      },
    });

    await expect(flowAPI.getFlowInfo('NonExistent')).rejects.toBeDefined();
  });

  it('should handle 500 errors', async () => {
    mockAxios.get.mockRejectedValueOnce({
      response: {
        status: 500,
        data: { detail: 'Internal server error' },
      },
    });

    await expect(flowAPI.listFlows()).rejects.toBeDefined();
  });
});

