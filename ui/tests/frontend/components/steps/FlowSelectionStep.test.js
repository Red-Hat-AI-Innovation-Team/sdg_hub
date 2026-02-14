// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for FlowSelectionStep - Flow selection with categories and search.
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API
jest.mock('../../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  flowAPI: {
    listFlowsWithDetails: jest.fn().mockResolvedValue([
      {
        name: 'Advanced Document Grounded QA Generation Flow',
        flow_id: 'small-rock-799',
        tags: ['question-generation', 'knowledge-extraction', 'qa-pairs'],
        description: 'Generate Q&A pairs from documents using multi-summary approach',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
      {
        name: 'Structured Insights Flow',
        flow_id: 'blue-sky-123',
        tags: ['text-analysis', 'extraction', 'insights'],
        description: 'Extract structured insights, keywords, and summaries from text',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
      {
        name: 'RAG Evaluation Pipeline',
        flow_id: 'green-tree-456',
        tags: ['evaluation', 'benchmark', 'rag'],
        description: 'Evaluate RAG system with groundedness and relevancy scoring',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
    ]),
    getFlowInfo: jest.fn().mockResolvedValue({
      name: 'Advanced Document Grounded QA Generation Flow',
      flow_id: 'small-rock-799',
      tags: ['question-generation'],
      blocks: [{ name: 'block1', type: 'LLMChatBlock' }],
    }),
    selectFlow: jest.fn().mockResolvedValue({ status: 'success' }),
    searchFlows: jest.fn().mockResolvedValue({ flows: [] }),
  },
  configAPI: {
    getCurrent: jest.fn().mockResolvedValue({}),
  },
  API_BASE_URL: 'http://localhost:8000',
}));

import FlowSelectionStep from '../../../../frontend/src/components/steps/FlowSelectionStep';

// Helper to render with act-wrapped async flushing
const renderAndWait = async (props) => {
  let result;
  await act(async () => {
    result = render(<FlowSelectionStep {...props} />);
  });
  // Flush pending async effects
  await act(async () => {
    await new Promise(r => setTimeout(r, 100));
  });
  return result;
};

describe('FlowSelectionStep', () => {
  const defaultProps = {
    selectedFlow: null,
    onFlowSelect: jest.fn(),
    importedConfig: null,
    initialCustomYaml: null,
    onError: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the flow selection interface', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText('Available Flows')).toBeInTheDocument();
  });

  it('loads and displays available flows in categories', async () => {
    await renderAndWait(defaultProps);
    // Flows are grouped by category
    expect(screen.getByText(/Knowledge Generation/i)).toBeInTheDocument();
  });

  it('displays flow categories', async () => {
    await renderAndWait(defaultProps);
    // Categories rendered based on flow tags
    expect(screen.getByText(/Knowledge Generation/i)).toBeInTheDocument();
    expect(screen.getByText(/Text Analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/Evaluation/i)).toBeInTheDocument();
  });

  it('has a search/filter input', async () => {
    await renderAndWait(defaultProps);
    const searchInput = screen.queryByRole('searchbox') ||
                        screen.queryByPlaceholderText(/search/i) ||
                        screen.queryByPlaceholderText(/filter/i);
    expect(searchInput).toBeTruthy();
  });

  it('shows flow count', async () => {
    await renderAndWait(defaultProps);
    // Text shows "X of Y flows" format
    const countText = screen.queryByText(/of.*flow/i);
    expect(countText).toBeTruthy();
  });

  it('shows Flow Details panel', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText('Flow Details')).toBeInTheDocument();
    expect(screen.getByText(/No flow selected/i)).toBeInTheDocument();
  });

  it('shows selected flow details when a flow is pre-selected', async () => {
    const selectedFlow = {
      name: 'Advanced Document Grounded QA Generation Flow',
      flow_id: 'small-rock-799',
      tags: ['question-generation'],
    };

    await renderAndWait({ ...defaultProps, selectedFlow });
    // Should not show "No flow selected"
    expect(screen.queryByText(/No flow selected/i)).not.toBeInTheDocument();
  });

  it('displays filter-by-tags section', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Filter by tags/i)).toBeInTheDocument();
  });

  it('handles API errors gracefully', async () => {
    const api = require('../../../../frontend/src/services/api');
    api.flowAPI.listFlowsWithDetails.mockRejectedValueOnce(new Error('API Error'));
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    await renderAndWait(defaultProps);

    // Should show error state or empty state, not crash
    expect(screen.getByText('Available Flows')).toBeInTheDocument();

    consoleSpy.mockRestore();
  });
});
