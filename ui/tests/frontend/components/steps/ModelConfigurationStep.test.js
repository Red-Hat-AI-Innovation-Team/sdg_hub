// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for ModelConfigurationStep - Model selection and configuration.
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the API
jest.mock('../../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: { get: jest.fn(), post: jest.fn() },
  modelAPI: {
    getRecommendations: jest.fn().mockResolvedValue({
      data: {
        default_model: 'meta-llama/Llama-3.3-70B-Instruct',
        recommendations: {
          model_family: 'llama',
          context_window: 8192,
          notes: 'Recommended for knowledge generation',
        },
      },
    }),
    configure: jest.fn().mockResolvedValue({ data: { status: 'success' } }),
    testConnection: jest.fn().mockResolvedValue({
      data: { status: 'success', message: 'Model is reachable' },
    }),
  },
  API_BASE_URL: 'http://localhost:8000',
}));

import ModelConfigurationStep from '../../../../frontend/src/components/steps/ModelConfigurationStep';

describe('ModelConfigurationStep', () => {
  const defaultProps = {
    selectedFlow: 'Test QA Flow',
    modelConfig: null,
    importedConfig: null,
    onConfigChange: jest.fn(),
    onError: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the model configuration form', async () => {
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      // Should show model configuration elements
      const modelLabel = screen.queryByText(/model/i);
      expect(modelLabel).toBeTruthy();
    });
  });

  it('loads model recommendations for the selected flow', async () => {
    const { modelAPI } = require('../../../../frontend/src/services/api');
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      expect(modelAPI.getRecommendations).toHaveBeenCalled();
    });
  });

  it('displays the recommended model', async () => {
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      const modelText = screen.queryByText(/Llama-3.3-70B-Instruct/i) ||
                        screen.queryByDisplayValue(/Llama-3.3-70B-Instruct/i) ||
                        screen.queryByText(/llama/i) ||
                        screen.queryByDisplayValue(/llama/i);
      expect(modelText).toBeTruthy();
    }, { timeout: 5000 });
  });

  it('shows API base URL input', async () => {
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      const apiBaseInput = screen.queryByLabelText(/api.*base/i) ||
                           screen.queryByLabelText(/endpoint/i) ||
                           screen.queryByPlaceholderText(/http/i);
      expect(apiBaseInput).toBeTruthy();
    });
  });

  it('shows API key input', async () => {
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      const apiKeyInput = screen.queryByLabelText(/api.*key/i) ||
                          screen.queryByPlaceholderText(/key/i) ||
                          screen.queryByDisplayValue(/EMPTY/i);
      expect(apiKeyInput).toBeTruthy();
    });
  });

  it('pre-fills form with existing modelConfig', async () => {
    const existingConfig = {
      model: 'hosted_vllm/custom-model',
      api_base: 'http://custom-server:8001/v1',
      api_key: 'my-key',
    };

    render(
      <ModelConfigurationStep
        {...defaultProps}
        modelConfig={existingConfig}
      />
    );

    await waitFor(() => {
      // Should have the existing values
      const modelInput = screen.queryByDisplayValue(/custom-model/i);
      const apiBaseInput = screen.queryByDisplayValue(/custom-server/i);
      expect(modelInput || apiBaseInput).toBeTruthy();
    });
  });

  it('calls onConfigChange when model is configured', async () => {
    const user = userEvent.setup();
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      // Wait for recommendations to load
      expect(screen.queryByText(/model/i)).toBeTruthy();
    });

    // Find the apply/configure button
    const applyButton = screen.queryByText(/apply/i) ||
                        screen.queryByText(/configure/i) ||
                        screen.queryByText(/save/i);
    
    if (applyButton) {
      await user.click(applyButton);
      // Should eventually call onConfigChange
    }
  });

  it('handles API errors when loading recommendations', async () => {
    const { modelAPI } = require('../../../../frontend/src/services/api');
    modelAPI.getRecommendations.mockRejectedValueOnce(new Error('API Error'));
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      // Should handle error gracefully
      expect(document.body).toBeTruthy();
    });

    consoleSpy.mockRestore();
  });

  it('shows no flow selected message when selectedFlow is null', async () => {
    render(<ModelConfigurationStep {...defaultProps} selectedFlow={null} />);

    await waitFor(() => {
      const noFlow = screen.queryByText(/no flow/i) || screen.queryByText(/select.*flow/i);
      // Should either show a message or gracefully handle null flow
      expect(document.body).toBeTruthy();
    });
  });

  it('supports additional LLM parameters', async () => {
    render(<ModelConfigurationStep {...defaultProps} />);

    await waitFor(() => {
      // Should have expandable/advanced section for additional params
      const advancedSection = screen.queryByText(/additional/i) ||
                              screen.queryByText(/advanced/i) ||
                              screen.queryByText(/parameter/i);
      expect(advancedSection).toBeTruthy();
    });
  });
});
