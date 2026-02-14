// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for ConfigurationTable component - Displays saved flow configurations.
 */

import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API
jest.mock('../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  runsAPI: {
    list: jest.fn().mockResolvedValue({ data: [] }),
  },
}));

import ConfigurationTable from '../../../frontend/src/components/configurations/ConfigurationTable';

const mockConfigurations = [
  {
    id: 'config-1',
    flow_name: 'Test QA Flow',
    flow_id: 'small-rock-799',
    model_configuration: { model: 'hosted_vllm/meta-llama/Llama-3.3-70B-Instruct' },
    dataset_configuration: { data_files: 'test.jsonl', num_samples: 10 },
    status: 'configured',
    created_at: '2026-02-01T10:00:00',
    tags: ['question-generation'],
  },
  {
    id: 'config-2',
    flow_name: 'Text Analysis Flow',
    flow_id: 'blue-sky-123',
    model_configuration: { model: 'hosted_vllm/granite-3b' },
    dataset_configuration: { data_files: 'data.csv', num_samples: 5 },
    status: 'configured',
    created_at: '2026-02-02T12:00:00',
    tags: ['text-analysis'],
  },
];

describe('ConfigurationTable', () => {
  const defaultProps = {
    configurations: mockConfigurations,
    selectedConfigs: [],
    onToggleSelection: jest.fn(),
    onToggleSelectAll: jest.fn(),
    onDryRun: jest.fn(),
    onGenerate: jest.fn(),
    onEdit: jest.fn(),
    onDelete: jest.fn(),
    onStop: jest.fn(),
    onClone: jest.fn(),
    isRunning: jest.fn().mockReturnValue(false),
    onFlowNameClick: jest.fn(),
    executionStates: {},
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a table with configurations', () => {
    render(<ConfigurationTable {...defaultProps} />);
    expect(screen.getByText('Test QA Flow')).toBeInTheDocument();
    expect(screen.getByText('Text Analysis Flow')).toBeInTheDocument();
  });

  it('displays model names for each configuration', () => {
    render(<ConfigurationTable {...defaultProps} />);
    // The model names should be visible (possibly truncated)
    expect(screen.getByText(/Llama-3.3-70B-Instruct/i)).toBeInTheDocument();
    expect(screen.getByText(/granite-3b/i)).toBeInTheDocument();
  });

  it('renders empty state when no configurations', () => {
    render(<ConfigurationTable {...defaultProps} configurations={[]} />);
    // Should not show table rows for configurations
    expect(screen.queryByText('Test QA Flow')).not.toBeInTheDocument();
  });

  it('calls onFlowNameClick when flow name is clicked', async () => {
    const user = userEvent.setup();
    render(<ConfigurationTable {...defaultProps} />);

    const flowLink = screen.getByText('Test QA Flow');
    await user.click(flowLink);

    expect(defaultProps.onFlowNameClick).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'config-1' })
    );
  });

  it('shows action buttons for each configuration', () => {
    render(<ConfigurationTable {...defaultProps} />);
    // Should have action menus (kebab / ellipsis buttons)
    const actionButtons = screen.getAllByRole('button');
    expect(actionButtons.length).toBeGreaterThan(0);
  });

  it('handles selection toggle for individual configs', async () => {
    const user = userEvent.setup();
    render(<ConfigurationTable {...defaultProps} />);

    // Find checkboxes
    const checkboxes = screen.getAllByRole('checkbox');
    if (checkboxes.length > 1) {
      await user.click(checkboxes[1]); // First row checkbox (index 0 may be select-all)
      expect(defaultProps.onToggleSelection).toHaveBeenCalled();
    }
  });

  it('renders table rows for each configuration', () => {
    render(<ConfigurationTable {...defaultProps} />);
    // The table should have rows for each config
    const rows = screen.getAllByRole('row');
    // Header row + 2 data rows
    expect(rows.length).toBeGreaterThanOrEqual(3);
  });

  it('shows execution state when a flow is running', () => {
    const executionStates = {
      'config-1': {
        isRunning: true,
        status: 'running',
        rawOutput: 'Processing block 2/5: LLMChatBlock',
      },
    };

    render(
      <ConfigurationTable
        {...defaultProps}
        executionStates={executionStates}
        isRunning={jest.fn().mockReturnValue(true)}
      />
    );

    // Should show running indicator (via Label or text)
    const runningText = screen.queryByText(/running/i) || screen.queryByText(/progress/i) || screen.queryByText(/block/i);
    expect(runningText).toBeTruthy();
  });

  it('shows completed state with output info', () => {
    const executionStates = {
      'config-1': {
        isRunning: false,
        status: 'completed',
        outputSamples: 50,
      },
    };

    render(
      <ConfigurationTable
        {...defaultProps}
        executionStates={executionStates}
      />
    );

    expect(screen.getByText(/completed/i)).toBeInTheDocument();
  });

  it('shows failed state', () => {
    const executionStates = {
      'config-1': {
        isRunning: false,
        status: 'failed',
        error: 'Model connection timeout',
      },
    };

    render(
      <ConfigurationTable
        {...defaultProps}
        executionStates={executionStates}
      />
    );

    expect(screen.getByText(/failed/i)).toBeInTheDocument();
  });
});
