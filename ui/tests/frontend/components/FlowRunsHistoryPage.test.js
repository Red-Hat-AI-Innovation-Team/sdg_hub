// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for FlowRunsHistoryPage component - Run history and status tracking.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API (data is inlined because jest.mock is hoisted before variable declarations)
jest.mock('../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  runsAPI: {
    // runsAPI.list() returns response.data which has { runs: [...] }
    list: jest.fn().mockResolvedValue({
      runs: [
        {
          run_id: 'run-1',
          config_id: 'config-1',
          flow_name: 'Document QA Flow',
          flow_type: 'existing',
          model_name: 'hosted_vllm/meta-llama/Llama-3.3-70B-Instruct',
          status: 'completed',
          start_time: '2026-02-08T10:00:00Z',
          end_time: '2026-02-08T10:05:30Z',
          duration_seconds: 330,
          input_samples: 10,
          output_samples: 50,
          output_file: 'outputs/run-1.jsonl',
        },
        {
          run_id: 'run-2',
          config_id: 'config-2',
          flow_name: 'Text Analysis Flow',
          flow_type: 'existing',
          model_name: 'hosted_vllm/granite-3b',
          status: 'failed',
          start_time: '2026-02-09T14:00:00Z',
          end_time: '2026-02-09T14:01:00Z',
          duration_seconds: 60,
          input_samples: 5,
          output_samples: null,
          error_message: 'Model connection timeout',
        },
      ],
    }),
    delete: jest.fn().mockResolvedValue({}),
  },
  savedConfigAPI: {
    // savedConfigAPI.list() returns response.data which has { configurations: [...] }
    list: jest.fn().mockResolvedValue({ configurations: [] }),
  },
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock NotificationContext
jest.mock('../../../frontend/src/contexts/NotificationContext', () => ({
  useNotifications: () => ({
    addSuccessNotification: jest.fn(),
    addErrorNotification: jest.fn(),
    addWarningNotification: jest.fn(),
    addInfoNotification: jest.fn(),
  }),
}));

import FlowRunsHistoryPage from '../../../frontend/src/components/FlowRunsHistoryPage';

// Helper for rendering with async flushing
const renderAndWait = async (props) => {
  let result;
  await act(async () => {
    result = render(<FlowRunsHistoryPage {...props} />);
  });
  await act(async () => {
    await new Promise(r => setTimeout(r, 200));
  });
  return result;
};

// Re-usable mock data for beforeEach
const freshMockResponse = () => ({
  runs: [
    {
      run_id: 'run-1', config_id: 'config-1', flow_name: 'Document QA Flow',
      flow_type: 'existing', model_name: 'hosted_vllm/meta-llama/Llama-3.3-70B-Instruct',
      status: 'completed', start_time: '2026-02-08T10:00:00Z', end_time: '2026-02-08T10:05:30Z',
      duration_seconds: 330, input_samples: 10, output_samples: 50, output_file: 'outputs/run-1.jsonl',
    },
    {
      run_id: 'run-2', config_id: 'config-2', flow_name: 'Text Analysis Flow',
      flow_type: 'existing', model_name: 'hosted_vllm/granite-3b',
      status: 'failed', start_time: '2026-02-09T14:00:00Z', end_time: '2026-02-09T14:01:00Z',
      duration_seconds: 60, input_samples: 5, output_samples: null, error_message: 'Model connection timeout',
    },
  ],
});

describe('FlowRunsHistoryPage', () => {
  const defaultProps = {
    executionStates: {},
    onUpdateExecutionState: jest.fn(),
    getExecutionState: jest.fn().mockReturnValue(null),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    const api = require('../../../frontend/src/services/api');
    api.runsAPI.list.mockResolvedValue(freshMockResponse());
    api.savedConfigAPI.list.mockResolvedValue({ configurations: [] });
  });

  it('renders the page title', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText('Flow Runs History')).toBeInTheDocument();
  });

  it('loads and displays run history', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText('Document QA Flow')).toBeInTheDocument();
    expect(screen.getByText('Text Analysis Flow')).toBeInTheDocument();
  });

  it('displays run status labels', async () => {
    await renderAndWait(defaultProps);
    // Multiple elements may match (summary stats + table rows)
    expect(screen.getAllByText(/completed/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/failed/i).length).toBeGreaterThan(0);
  });

  it('displays model names', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Llama-3.3-70B/i)).toBeInTheDocument();
  });

  it('shows empty state when no runs exist', async () => {
    const api = require('../../../frontend/src/services/api');
    api.runsAPI.list.mockResolvedValueOnce({ runs: [] });

    await renderAndWait(defaultProps);
    const emptyText = screen.queryByText(/no.*run/i) || screen.queryByText(/empty/i);
    expect(emptyText).toBeTruthy();
  });

  it('has search/filter functionality', async () => {
    await renderAndWait(defaultProps);
    const searchInput = screen.queryByRole('searchbox') ||
                        screen.queryByPlaceholderText(/search/i) ||
                        screen.queryByRole('textbox');
    expect(searchInput).toBeTruthy();
  });

  it('handles API errors during load', async () => {
    const api = require('../../../frontend/src/services/api');
    api.runsAPI.list.mockRejectedValueOnce(new Error('Network Error'));
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    await renderAndWait(defaultProps);
    // Multiple error text elements may appear (alert title + body)
    const errorElements = screen.queryAllByText(/error/i);
    expect(errorElements.length).toBeGreaterThan(0);

    consoleSpy.mockRestore();
  });
});
