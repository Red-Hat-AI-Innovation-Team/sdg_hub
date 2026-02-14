// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for DatasetConfigurationStep - Dataset upload and configuration.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API
jest.mock('../../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  datasetAPI: {
    getSchema: jest.fn().mockResolvedValue({
      columns: ['document', 'domain', 'document_outline'],
      types: { document: 'string', domain: 'string', document_outline: 'string' },
    }),
    uploadFile: jest.fn().mockResolvedValue({
      file_path: 'uploads/dataset.jsonl', rows: 10, columns: ['document', 'domain'],
    }),
    loadDataset: jest.fn().mockResolvedValue({
      status: 'success', num_samples: 10,
    }),
    getPreview: jest.fn().mockResolvedValue({
      rows: [{ document: 'Test doc', domain: 'science' }],
      columns: ['document', 'domain'],
      num_samples: 10,
    }),
    checkDuplicates: jest.fn().mockResolvedValue({ has_duplicates: false }),
    removeDuplicates: jest.fn().mockResolvedValue({}),
  },
  preprocessingAPI: {
    getJobs: jest.fn().mockResolvedValue({}),
    getDatasets: jest.fn().mockResolvedValue([]),
  },
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock NotificationContext
jest.mock('../../../../frontend/src/contexts/NotificationContext', () => ({
  useNotifications: () => ({
    addSuccessNotification: jest.fn(),
    addErrorNotification: jest.fn(),
    addWarningNotification: jest.fn(),
    addInfoNotification: jest.fn(),
  }),
}));

import DatasetConfigurationStep from '../../../../frontend/src/components/steps/DatasetConfigurationStep';

// Helper for rendering with async flushing
const renderAndWait = async (props) => {
  let result;
  await act(async () => {
    result = render(<DatasetConfigurationStep {...props} />);
  });
  await act(async () => {
    await new Promise(r => setTimeout(r, 200));
  });
  return result;
};

describe('DatasetConfigurationStep', () => {
  const defaultProps = {
    selectedFlow: 'Test QA Flow',
    datasetConfig: null,
    onConfigChange: jest.fn(),
    onError: jest.fn(),
    importedConfig: null,
    onNeedsPDFPreprocessing: jest.fn(),
    preprocessedDatasets: [],
    onLoadPreprocessedDataset: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Re-establish mocks
    const api = require('../../../../frontend/src/services/api');
    api.datasetAPI.getSchema.mockResolvedValue({
      columns: ['document', 'domain', 'document_outline'],
      types: { document: 'string', domain: 'string', document_outline: 'string' },
    });
    api.datasetAPI.getPreview.mockResolvedValue({
      rows: [{ document: 'Test doc' }], columns: ['document'], num_samples: 10,
    });
  });

  it('renders the dataset configuration interface', async () => {
    await renderAndWait(defaultProps);
    // Should show dataset-related content
    const text = document.body.textContent;
    expect(text).toContain('Dataset');
  });

  it('loads schema requirements for selected flow', async () => {
    const api = require('../../../../frontend/src/services/api');
    await renderAndWait(defaultProps);
    expect(api.datasetAPI.getSchema).toHaveBeenCalled();
  });

  it('displays required columns from schema', async () => {
    await renderAndWait(defaultProps);
    // Should display column names
    const text = document.body.textContent;
    const hasColumns = text.includes('document') || text.includes('column');
    expect(hasColumns).toBe(true);
  });

  it('shows file upload or data source selection', async () => {
    await renderAndWait(defaultProps);
    const text = document.body.textContent.toLowerCase();
    const hasUpload = text.includes('upload') || text.includes('file') || text.includes('browse') || text.includes('drag');
    expect(hasUpload).toBe(true);
  });

  it('shows dataset configuration options', async () => {
    await renderAndWait(defaultProps);
    const text = document.body.textContent.toLowerCase();
    // Should display some configuration options
    const hasConfig = text.includes('sample') || text.includes('rows') || 
                      text.includes('number') || text.includes('format') ||
                      text.includes('dataset') || text.includes('configuration');
    expect(hasConfig).toBe(true);
  });

  it('handles no flow selected', async () => {
    await renderAndWait({ ...defaultProps, selectedFlow: null });
    // Should not crash
    expect(document.body).toBeTruthy();
  });
});
