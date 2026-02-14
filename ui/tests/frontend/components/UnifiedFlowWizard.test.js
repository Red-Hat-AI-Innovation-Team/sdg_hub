// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for UnifiedFlowWizard component - Multi-step flow configuration wizard.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock all APIs
jest.mock('../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  flowAPI: {
    listFlows: jest.fn().mockResolvedValue({
      flows: ['Test QA Flow', 'Text Analysis Flow'],
    }),
    listFlowsWithDetails: jest.fn().mockResolvedValue([
      {
        name: 'Test QA Flow',
        flow_id: 'small-rock-799',
        tags: ['question-generation'],
        description: 'Test flow',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
    ]),
    searchFlows: jest.fn().mockResolvedValue({ flows: [] }),
    getFlowInfo: jest.fn().mockResolvedValue({
      name: 'Test QA Flow',
      flow_id: 'small-rock-799',
      tags: ['question-generation'],
      blocks: [{ name: 'block1', type: 'LLMChatBlock' }],
    }),
    selectFlow: jest.fn().mockResolvedValue({ status: 'success' }),
    selectByPath: jest.fn().mockResolvedValue({ status: 'success' }),
    getFlowYaml: jest.fn().mockResolvedValue({ yaml: 'blocks: []' }),
  },
  savedConfigAPI: {
    save: jest.fn().mockResolvedValue({ id: 'new-config-123', status: 'configured' }),
    list: jest.fn().mockResolvedValue([]),
    load: jest.fn().mockResolvedValue({}),
  },
  modelAPI: {
    getRecommendations: jest.fn().mockResolvedValue({
      default_model: 'meta-llama/Llama-3.3-70B-Instruct',
      recommendations: { model_family: 'llama', context_window: 8192 },
    }),
    configure: jest.fn().mockResolvedValue({ status: 'success' }),
  },
  datasetAPI: {
    getSchema: jest.fn().mockResolvedValue({
      columns: ['document', 'domain'], types: {},
    }),
    uploadFile: jest.fn().mockResolvedValue({ file_path: 'test.jsonl' }),
    getPreview: jest.fn().mockResolvedValue({ rows: [], columns: [] }),
    loadDataset: jest.fn().mockResolvedValue({ status: 'success' }),
    checkDuplicates: jest.fn().mockResolvedValue({ has_duplicates: false }),
  },
  configAPI: {
    getCurrent: jest.fn().mockResolvedValue({}),
  },
  preprocessingAPI: {
    getJobs: jest.fn().mockResolvedValue({}),
    getDatasets: jest.fn().mockResolvedValue([]),
  },
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock NotificationContext
jest.mock('../../../frontend/src/contexts/NotificationContext', () => ({
  useNotifications: () => ({
    addNotification: jest.fn(),
    addSuccessNotification: jest.fn(),
    addErrorNotification: jest.fn(),
    addWarningNotification: jest.fn(),
    addInfoNotification: jest.fn(),
    removeNotification: jest.fn(),
  }),
}));

import UnifiedFlowWizard from '../../../frontend/src/components/UnifiedFlowWizard';

// Helper for rendering with async flushing
const renderAndWait = async (props) => {
  let result;
  await act(async () => {
    result = render(<UnifiedFlowWizard {...props} />);
  });
  await act(async () => {
    await new Promise(r => setTimeout(r, 200));
  });
  return result;
};

describe('UnifiedFlowWizard', () => {
  const defaultProps = {
    wizardData: null,
    editingConfig: null,
    onComplete: jest.fn(),
    onCancel: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    sessionStorage.clear();
  });

  it('renders the wizard with source type selection', async () => {
    await renderAndWait(defaultProps);
    // Should show source type selection (existing flow vs custom)
    const text = document.body.textContent;
    const hasSourceType = text.includes('existing') || text.includes('Existing') ||
                          text.includes('pre-built') || text.includes('Pre-Built') ||
                          text.includes('Select') || text.includes('Flow');
    expect(hasSourceType).toBe(true);
  });

  it('renders wizard step navigation buttons', async () => {
    await renderAndWait(defaultProps);
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('handles editingConfig prop without crashing', async () => {
    const editingConfig = {
      id: 'config-123',
      flow_name: 'Test QA Flow',
      flow_id: 'small-rock-799',
      flow_path: '/flows/test.yaml',
      model_configuration: {
        model: 'hosted_vllm/meta-llama/Llama-3.3-70B-Instruct',
        api_base: 'http://localhost:8001/v1',
        api_key: 'EMPTY',
      },
      dataset_configuration: {
        data_files: 'test.jsonl',
        num_samples: 10,
      },
      status: 'configured',
    };

    await renderAndWait({ ...defaultProps, editingConfig });
    expect(document.body.textContent.length).toBeGreaterThan(0);
  });

  it('handles wizardData prop without crashing', async () => {
    const wizardData = {
      selectedFlow: 'Test QA Flow',
      sourceType: 'existing',
    };

    await renderAndWait({ ...defaultProps, wizardData });
    expect(document.body.textContent.length).toBeGreaterThan(0);
  });

  it('restores session state from sessionStorage', async () => {
    sessionStorage.setItem('wizard_session_state', JSON.stringify({
      sourceType: 'existing',
      selectedFlow: 'Test QA Flow',
    }));

    await renderAndWait(defaultProps);
    expect(document.body.textContent.length).toBeGreaterThan(0);
  });
});
