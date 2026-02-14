// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for HomeDashboard component - Landing page with flow categories.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API
jest.mock('../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  flowAPI: {
    listFlowsWithDetails: jest.fn().mockResolvedValue([
      {
        name: 'Document Grounded QA Flow',
        flow_id: 'small-rock-799',
        tags: ['question-generation', 'knowledge-extraction'],
        description: 'Generate Q&A pairs from documents',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
      {
        name: 'Structured Insights Flow',
        flow_id: 'blue-sky-123',
        tags: ['text-analysis', 'extraction'],
        description: 'Extract structured insights from text',
        model_recommendations: { default_model: 'meta-llama/Llama-3.3-70B-Instruct' },
      },
    ]),
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

import HomeDashboard from '../../../frontend/src/components/HomeDashboard';

// Helper for rendering with async flushing
const renderAndWait = async (props) => {
  let result;
  await act(async () => {
    result = render(<HomeDashboard {...props} />);
  });
  await act(async () => {
    await new Promise(r => setTimeout(r, 200));
  });
  return result;
};

describe('HomeDashboard', () => {
  const defaultProps = {
    onNavigate: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the welcome title', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Welcome to SDG Hub/i)).toBeInTheDocument();
  });

  it('renders description text', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Synthetic Data Generation/i)).toBeInTheDocument();
  });

  it('renders quick start section', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Quick Start/i)).toBeInTheDocument();
    expect(screen.getByText(/Create New Flow/i)).toBeInTheDocument();
  });

  it('renders UI pages guide', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Home/)).toBeInTheDocument();
    expect(screen.getByText(/Dashboard/)).toBeInTheDocument();
    expect(screen.getByText(/Data Generation Flows/i)).toBeInTheDocument();
    expect(screen.getByText(/Run History/i)).toBeInTheDocument();
  });

  it('renders key capabilities section', async () => {
    await renderAndWait(defaultProps);
    expect(screen.getByText(/Key Capabilities/i)).toBeInTheDocument();
    expect(screen.getByText(/Pre-built generation pipelines/i)).toBeInTheDocument();
    expect(screen.getByText(/Custom flow builder/i)).toBeInTheDocument();
  });

  it('displays flow categories after loading', async () => {
    await renderAndWait(defaultProps);
    const knowledgeSection = screen.queryByText(/Knowledge Generation/i);
    const textSection = screen.queryByText(/Text Analysis/i);
    expect(knowledgeSection || textSection).toBeTruthy();
  });

  it('calls onNavigate when Create New Flow is clicked', async () => {
    const user = userEvent.setup();
    await renderAndWait(defaultProps);

    const createButton = screen.getByText(/Create New Flow/i);
    await user.click(createButton);

    expect(defaultProps.onNavigate).toHaveBeenCalled();
  });

  it('handles API errors gracefully', async () => {
    const api = require('../../../frontend/src/services/api');
    api.flowAPI.listFlowsWithDetails.mockRejectedValueOnce(new Error('API Error'));
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    await renderAndWait(defaultProps);

    // Should not crash - still shows static content
    expect(screen.getByText(/Welcome to SDG Hub/i)).toBeInTheDocument();

    consoleSpy.mockRestore();
  });
});
