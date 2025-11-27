import React, { useState, useEffect } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Form,
  FormGroup,
  TextInput,
  MenuToggle,
  Select,
  SelectOption,
  SelectList,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  Grid,
  GridItem,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  Chip,
  ChipGroup,
  ExpandableSection,
  Tooltip,
  Flex,
  FlexItem,
} from '@patternfly/react-core';
import { CheckCircleIcon, InfoCircleIcon } from '@patternfly/react-icons';
import { modelAPI } from '../../services/api';

/**
 * Model Configuration Step Component
 * 
 * Allows users to:
 * - View recommended models for the selected flow
 * - Configure model settings (model, api_base, api_key)
 * - Add additional LLM parameters
 * - Test the configuration
 */
const ModelConfigurationStep = ({ selectedFlow, modelConfig, importedConfig, onConfigChange, onError }) => {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isConfigured, setIsConfigured] = useState(false);
  const [isConfiguring, setIsConfiguring] = useState(false);

  // Form state
  const [model, setModel] = useState('');
  const [apiBase, setApiBase] = useState('');
  const [apiKey, setApiKey] = useState('EMPTY');
  const [additionalParams, setAdditionalParams] = useState({
    temperature: '',
    max_tokens: '',
    top_p: '',
    save_freq: '',
    n: '',
    timeout: '',
    num_retries: '',
  });

  // Pre-fill form with existing modelConfig or imported configuration
  useEffect(() => {
    const configToUse = importedConfig || modelConfig;
    
    if (configToUse && Object.keys(configToUse).length > 0) {
      if (configToUse.model) setModel(configToUse.model);
      if (configToUse.api_base) setApiBase(configToUse.api_base);
      if (configToUse.api_key) setApiKey(configToUse.api_key);
      if (configToUse.additional_params) {
        setAdditionalParams({
          temperature: configToUse.additional_params.temperature || '',
          max_tokens: configToUse.additional_params.max_tokens || '',
          top_p: configToUse.additional_params.top_p || '',
          save_freq: configToUse.additional_params.save_freq || '',
          n: configToUse.additional_params.n || '',
          timeout: configToUse.additional_params.timeout || '',
          num_retries: configToUse.additional_params.num_retries || '',
        });
      }
      setIsConfigured(true);
    }
  }, [importedConfig, modelConfig]);

  // UI state
  const [isModelSelectOpen, setIsModelSelectOpen] = useState(false);
  const [isAdvancedExpanded, setIsAdvancedExpanded] = useState(false);

  /**
   * Load model recommendations when flow is selected
   */
  useEffect(() => {
    if (selectedFlow) {
      loadRecommendations();
    }
  }, [selectedFlow]);

  /**
   * Don't pre-fill model - let user choose from dropdown or type custom
   * Default model is shown as first option in the dropdown
   */

  /**
   * Load model recommendations from API or use selectedFlow's recommendations
   */
  const loadRecommendations = async () => {
    try {
      setLoading(true);
      
      // If selectedFlow has recommended_models, use those (for custom flows)
      if (selectedFlow?.recommended_models) {
        setRecommendations(selectedFlow.recommended_models);
      } else {
        // Otherwise load from backend API (for existing flows)
        const data = await modelAPI.getRecommendations();
        setRecommendations(data);
      }
    } catch (error) {
      // Don't show error for custom flows - they may not have backend flow
      console.warn('Could not load recommendations from backend, using defaults:', error.message);
      // Use empty recommendations as fallback
      setRecommendations({
        default: '',
        compatible: [],
        experimental: []
      });
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle model configuration submission
   */
  const handleConfigure = async () => {
    try {
      setIsConfiguring(true);

      // Build configuration object
      const config = {
        model,
        api_base: apiBase,
        api_key: apiKey,
        additional_params: {},
      };

      // Add non-empty additional params
      if (additionalParams.temperature) {
        config.additional_params.temperature = parseFloat(additionalParams.temperature);
      }
      if (additionalParams.max_tokens) {
        config.additional_params.max_tokens = parseInt(additionalParams.max_tokens, 10);
      }
      if (additionalParams.top_p) {
        config.additional_params.top_p = parseFloat(additionalParams.top_p);
      }
      if (additionalParams.save_freq) {
        config.additional_params.save_freq = parseInt(additionalParams.save_freq, 10);
      }
      if (additionalParams.n) {
        config.additional_params.n = parseInt(additionalParams.n, 10);
      }
      if (additionalParams.timeout) {
        config.additional_params.timeout = parseFloat(additionalParams.timeout);
      }
      if (additionalParams.num_retries) {
        config.additional_params.num_retries = parseInt(additionalParams.num_retries, 10);
      }

      // Update parent state first (saves draft)
      onConfigChange(config);
      
      // Send to API only if we have a backend flow selected (existing flows)
      // For custom flows, skip this step as there's no backend flow yet
      try {
        await modelAPI.configure(config);
      } catch (apiError) {
        // If API call fails (e.g., custom flow with no backend flow selected), that's okay
        // The config is still saved to parent state
        console.warn('Model API configure failed (expected for custom flows):', apiError.message);
      }
      
      setIsConfigured(true);

    } catch (error) {
      onError('Failed to configure model: ' + error.message);
    } finally {
      setIsConfiguring(false);
    }
  };

  /**
   * Check if form is valid
   */
  const isFormValid = () => {
    return model && apiBase && apiKey;
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem' }}>
        <Spinner size="xl" />
        <div style={{ marginTop: '1rem' }}>Loading model recommendations...</div>
      </div>
    );
  }

  if (!selectedFlow) {
    return (
      <Alert
        variant={AlertVariant.warning}
        isInline
        title="No flow selected"
      >
        Please select a flow in the previous step before configuring the model.
      </Alert>
    );
  }

  return (
    <Grid hasGutter style={{ height: '100%' }}>
      {/* Import Success Indicator */}
      {importedConfig && (
        <GridItem span={12}>
          <Alert
            variant={AlertVariant.success}
            isInline
            title="Model configuration loaded from import"
          >
            <p>
              ✅ Model settings have been pre-filled: <strong>{importedConfig.model}</strong>
            </p>
          </Alert>
        </GridItem>
      )}

      {/* Left Panel - Configuration Form */}
      <GridItem span={7} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Model Configuration
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            <Form>
              {/* Model Selection */}
              <FormGroup label="Model" isRequired fieldId="model-select">
                <TextInput
                  isRequired
                  type="text"
                  id="model-select"
                  value={model}
                  onChange={(event, value) => setModel(value)}
                  placeholder="Type model name or select from suggestions below..."
                  list="model-suggestions"
                />
                <datalist id="model-suggestions">
                  {/* Default Model - shown first */}
                  {recommendations?.default_model && (
                    <option value={`hosted_vllm/${recommendations.default_model}`}>
                      {recommendations.default_model} (Default - Recommended)
                    </option>
                  )}
                  {/* Compatible Models */}
                  {recommendations?.recommendations?.compatible?.map((rec) => (
                    <option key={rec} value={`hosted_vllm/${rec}`}>
                      {rec} (Compatible)
                    </option>
                  ))}
                  {/* Experimental Models */}
                  {recommendations?.recommendations?.experimental?.map((rec) => (
                    <option key={rec} value={`hosted_vllm/${rec}`}>
                      {rec} (Experimental)
                    </option>
                  ))}
                  {/* Common alternatives */}
                  <option value="openai/gpt-4o">OpenAI GPT-4o</option>
                  <option value="openai/gpt-4o-mini">OpenAI GPT-4o-mini</option>
                  <option value="anthropic/claude-3-5-sonnet-20241022">Anthropic Claude 3.5 Sonnet</option>
                </datalist>
                <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginTop: '0.5rem' }}>
                  💡 Start typing to see suggestions, or enter any model name in LiteLLM format (e.g., hosted_vllm/model-name, openai/gpt-4o)
                </div>
              </FormGroup>

              {/* API Base URL */}
              <FormGroup label="API Base URL" isRequired fieldId="api-base">
                <TextInput
                  isRequired
                  type="text"
                  id="api-base"
                  name="api-base"
                  value={apiBase}
                  onChange={(event, value) => setApiBase(value)}
                  placeholder="http://localhost:8000/v1"
                />
              </FormGroup>

              {/* API Key */}
              <FormGroup 
                label="API Key" 
                isRequired 
                fieldId="api-key"
                helperText={
                  <span>
                    🔐 <strong>Security Tip:</strong> Use <code>env:VARIABLE_NAME</code> to reference environment variables (e.g., <code>env:OPENAI_API_KEY</code>).
                    <br />Or enter <code>EMPTY</code> for local models without authentication.
                  </span>
                }
              >
                <TextInput
                  isRequired
                  type={apiKey?.startsWith('env:') ? 'text' : 'password'}
                  id="api-key"
                  name="api-key"
                  value={apiKey}
                  onChange={(event, value) => setApiKey(value)}
                  placeholder="Enter API key, 'EMPTY', or 'env:VARIABLE_NAME'"
                />
              </FormGroup>

              {/* Advanced Parameters */}
              <ExpandableSection
                toggleText="Advanced Parameters"
                isExpanded={isAdvancedExpanded}
                onToggle={() => setIsAdvancedExpanded(!isAdvancedExpanded)}
              >
                <Grid hasGutter>
                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="temperature"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Temperature</FlexItem>
                          <FlexItem>
                            <Tooltip content="Controls randomness in outputs. Lower values (0.0) make responses more focused, higher values (1.0) make them more creative.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="temperature"
                        value={additionalParams.temperature}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, temperature: value })
                        }
                        placeholder="0.7"
                        step="0.1"
                        min="0"
                        max="2"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="max-tokens"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Max Tokens</FlexItem>
                          <FlexItem>
                            <Tooltip content="Maximum number of tokens to generate in the response. Higher values allow longer outputs but increase cost and latency.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="max-tokens"
                        value={additionalParams.max_tokens}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, max_tokens: value })
                        }
                        placeholder="2048"
                        min="1"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="top-p"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Top P</FlexItem>
                          <FlexItem>
                            <Tooltip content="Nucleus sampling: only consider tokens with cumulative probability up to this value. Lower values make output more focused.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="top-p"
                        value={additionalParams.top_p}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, top_p: value })
                        }
                        placeholder="1.0"
                        step="0.1"
                        min="0"
                        max="1"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="save-freq"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Save Frequency</FlexItem>
                          <FlexItem>
                            <Tooltip content="Number of samples to process before saving a checkpoint. Lower values save more often but may slow execution slightly.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="save-freq"
                        value={additionalParams.save_freq}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, save_freq: value })
                        }
                        placeholder="10"
                        min="1"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="n"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>N (Completions)</FlexItem>
                          <FlexItem>
                            <Tooltip content="Number of completions to generate for each prompt. Useful for getting multiple variations of the same output.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="n"
                        value={additionalParams.n}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, n: value })
                        }
                        placeholder="1"
                        min="1"
                        max="10"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="timeout"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Timeout (seconds)</FlexItem>
                          <FlexItem>
                            <Tooltip content="Maximum time to wait for a response from the LLM API before timing out the request.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="timeout"
                        value={additionalParams.timeout}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, timeout: value })
                        }
                        placeholder="120"
                        min="1"
                      />
                    </FormGroup>
                  </GridItem>

                  <GridItem span={6}>
                    <FormGroup 
                      fieldId="num-retries"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Num Retries</FlexItem>
                          <FlexItem>
                            <Tooltip content="Number of times to retry a failed API request before giving up. Helps handle transient errors.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <TextInput
                        type="number"
                        id="num-retries"
                        value={additionalParams.num_retries}
                        onChange={(event, value) =>
                          setAdditionalParams({ ...additionalParams, num_retries: value })
                        }
                        placeholder="3"
                        min="0"
                        max="10"
                      />
                    </FormGroup>
                  </GridItem>
                </Grid>
              </ExpandableSection>

              {/* Configure Button */}
              <Button
                variant="primary"
                onClick={handleConfigure}
                isDisabled={!isFormValid()}
                isLoading={isConfiguring}
                style={{ marginTop: '1rem' }}
              >
                {isConfigured ? 'Update Configuration' : 'Apply Configuration'}
              </Button>

              {(isConfigured || (modelConfig && modelConfig.model)) && (
                <Alert
                  variant={AlertVariant.success}
                  isInline
                  title="Configuration applied"
                  style={{ marginTop: '1rem' }}
                />
              )}
            </Form>
          </CardBody>
        </Card>
      </GridItem>

      {/* Right Panel - Recommendations */}
      <GridItem span={5} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Recommendations
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            {recommendations && (
              <DescriptionList isHorizontal>
                <DescriptionListGroup>
                  <DescriptionListTerm>Default Model</DescriptionListTerm>
                  <DescriptionListDescription>
                    <code>{recommendations.default_model || 'N/A'}</code>
                  </DescriptionListDescription>
                </DescriptionListGroup>

                {recommendations.recommendations?.compatible?.length > 0 && (
                  <DescriptionListGroup>
                    <DescriptionListTerm>Compatible Models</DescriptionListTerm>
                    <DescriptionListDescription>
                      <ChipGroup>
                        {recommendations.recommendations.compatible.map((model) => (
                          <Chip key={model} isReadOnly onClick={() => setModel(`hosted_vllm/${model}`)}>
                            {model}
                          </Chip>
                        ))}
                      </ChipGroup>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                )}

                {recommendations.recommendations?.experimental?.length > 0 && (
                  <DescriptionListGroup>
                    <DescriptionListTerm>Experimental Models</DescriptionListTerm>
                    <DescriptionListDescription>
                      <ChipGroup>
                        {recommendations.recommendations.experimental.map((model) => (
                          <Chip key={model} isReadOnly>
                            {model}
                          </Chip>
                        ))}
                      </ChipGroup>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                )}

                <DescriptionListGroup>
                  <DescriptionListTerm>Requires Configuration</DescriptionListTerm>
                  <DescriptionListDescription>
                    {recommendations.requires_config ? 'Yes' : 'No'}
                  </DescriptionListDescription>
                </DescriptionListGroup>
              </DescriptionList>
            )}

            <div style={{ marginTop: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
              <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
                Quick Setup for Local vLLM
              </Title>
              <div style={{ fontSize: '0.875rem' }}>
                <p>
                  <strong>Model:</strong> <code>hosted_vllm/your-model-name</code>
                </p>
                <p>
                  <strong>API Base:</strong> <code>http://localhost:8000/v1</code>
                </p>
                <p>
                  <strong>API Key:</strong> <code>EMPTY</code>
                </p>
              </div>
            </div>
          </CardBody>
        </Card>
      </GridItem>
    </Grid>
  );
};

export default ModelConfigurationStep;

