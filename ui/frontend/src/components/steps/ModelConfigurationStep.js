import React, { useState, useEffect, useRef } from 'react';
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
  Slider,
  FormHelperText,
  HelperText,
  HelperTextItem,
  CodeBlock,
  CodeBlockCode,
  Modal,
  ModalVariant,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  InfoCircleIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  TimesCircleIcon,
} from '@patternfly/react-icons';
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

  // Model name validation state
  const [modelNameWarning, setModelNameWarning] = useState(null);

  // Track whether the user manually cleared the model field
  const userClearedModel = useRef(false);

  // Test connection state
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [showTestModal, setShowTestModal] = useState(false);

  // Slider values (separate from form state for controlled display)
  const [temperatureSlider, setTemperatureSlider] = useState(0.7);
  const [topPSlider, setTopPSlider] = useState(1.0);

  /**
   * Load model recommendations when flow is selected
   */
  useEffect(() => {
    if (selectedFlow) {
      loadRecommendations();
    }
  }, [selectedFlow]);

  // Reset the "user cleared" flag when a different flow is selected
  useEffect(() => {
    userClearedModel.current = false;
  }, [selectedFlow?.name]);

  /**
   * Auto-fill default model when flow is selected and has recommendations
   */
  useEffect(() => {
    // Only auto-fill if model is not already set, user hasn't manually cleared it,
    // and we have recommendations from selectedFlow
    if (!model && !userClearedModel.current && selectedFlow?.recommended_models?.default && !importedConfig && !modelConfig?.model) {
      const defaultModel = `hosted_vllm/${selectedFlow.recommended_models.default}`;
      setModel(defaultModel);
      // Also set default API base for local vLLM
      if (!apiBase) {
        setApiBase('http://localhost:8001/v1');
      }
    }
  }, [selectedFlow, model, importedConfig, modelConfig, apiBase]);

  /**
   * Sync slider values with additionalParams when they change externally (e.g., imported config)
   */
  useEffect(() => {
    if (additionalParams.temperature) {
      const temp = parseFloat(additionalParams.temperature);
      if (!isNaN(temp)) setTemperatureSlider(temp);
    }
    if (additionalParams.top_p) {
      const topP = parseFloat(additionalParams.top_p);
      if (!isNaN(topP)) setTopPSlider(topP);
    }
  }, [additionalParams.temperature, additionalParams.top_p]);

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
   * Validate model name and return warning if issues found
   */
  const validateModelName = (name) => {
    if (!name) return null;
    
    const warnings = [];
    
    // Check for spaces (common typo)
    if (name.includes(' ')) {
      warnings.push('Model name contains spaces - this may cause issues');
    }
    
    // Check for common format issues
    if (!name.includes('/') && !name.startsWith('gpt-')) {
      warnings.push('Model should be in format "provider/model-name" (e.g., hosted_vllm/llama-3)');
    }
    
    // Check for double slashes
    if (name.includes('//')) {
      warnings.push('Model name contains double slashes');
    }
    
    // Check for trailing/leading whitespace
    if (name !== name.trim()) {
      warnings.push('Model name has leading or trailing whitespace');
    }
    
    // Check for common typos in providers
    const commonProviders = ['hosted_vllm', 'openai', 'anthropic', 'azure', 'together', 'anyscale'];
    const provider = name.split('/')[0]?.toLowerCase();
    if (provider && !commonProviders.includes(provider) && !name.startsWith('gpt-')) {
      warnings.push(`Provider "${provider}" is not commonly used - verify it's correct`);
    }
    
    return warnings.length > 0 ? warnings : null;
  };

  /**
   * Handle model name change with validation
   */
  const handleModelChange = (event, value) => {
    setModel(value);
    // Track if user deliberately cleared the field (prevents auto-fill from overriding)
    userClearedModel.current = !value;
    const warnings = validateModelName(value);
    setModelNameWarning(warnings);
    // Clear test result when model changes
    setTestResult(null);
  };

  /**
   * Test the model connection with a simple prompt
   */
  const handleTestConnection = async () => {
    if (!model || !apiBase) {
      setTestResult({
        success: false,
        error: 'Model and API Base URL are required',
      });
      return;
    }
    
    setIsTestingConnection(true);
    setTestResult(null);
    setShowTestModal(true);
    
    try {
      const result = await modelAPI.testConnection({
        model,
        api_base: apiBase,
        api_key: apiKey,
        test_prompt: 'What is the capital of France? Answer in one word.',
      });
      
      setTestResult({
        success: result.success,
        response: result.response,
        latency: result.latency_ms,
        error: result.error,
      });
    } catch (error) {
      setTestResult({
        success: false,
        error: error.response?.data?.detail || error.message || 'Connection failed',
      });
    } finally {
      setIsTestingConnection(false);
    }
  };

  /**
   * Handle temperature slider change
   */
  const handleTemperatureChange = (value) => {
    setTemperatureSlider(value);
    setAdditionalParams(prev => ({ ...prev, temperature: value.toString() }));
  };

  /**
   * Handle top-p slider change
   */
  const handleTopPChange = (value) => {
    setTopPSlider(value);
    setAdditionalParams(prev => ({ ...prev, top_p: value.toString() }));
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
                  onChange={handleModelChange}
                  placeholder="e.g., hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
                  list="model-suggestions"
                  validated={modelNameWarning ? 'warning' : 'default'}
                />
                <datalist id="model-suggestions">
                  {/* Default Model from selected flow - shown first */}
                  {selectedFlow?.recommended_models?.default && (
                    <option value={`hosted_vllm/${selectedFlow.recommended_models.default}`}>
                      {selectedFlow.recommended_models.default} (Default - Recommended)
                    </option>
                  )}
                  {/* Compatible Models from selected flow */}
                  {selectedFlow?.recommended_models?.compatible?.map((rec) => (
                    <option key={rec} value={`hosted_vllm/${rec}`}>
                      {rec} (Compatible)
                    </option>
                  ))}
                  {/* Experimental Models from selected flow */}
                  {selectedFlow?.recommended_models?.experimental?.map((rec) => (
                    <option key={rec} value={`hosted_vllm/${rec}`}>
                      {rec} (Experimental)
                    </option>
                  ))}
                  {/* Common alternatives */}
                  <option value="openai/gpt-4o">OpenAI GPT-4o</option>
                  <option value="openai/gpt-4o-mini">OpenAI GPT-4o-mini</option>
                  <option value="anthropic/claude-3-5-sonnet-20241022">Anthropic Claude 3.5 Sonnet</option>
                </datalist>
                
                {/* Model Name Warnings */}
                {modelNameWarning && (
                  <FormHelperText>
                    <HelperText>
                      {modelNameWarning.map((warning, idx) => (
                        <HelperTextItem key={idx} variant="warning" icon={<ExclamationTriangleIcon />}>
                          {warning}
                        </HelperTextItem>
                      ))}
                    </HelperText>
                  </FormHelperText>
                )}
                
                <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginTop: '0.5rem' }}>
                  💡 Format: <code>provider/model-name</code> (e.g., hosted_vllm/llama-3, openai/gpt-4o)
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
                  {/* Temperature Slider */}
                  <GridItem span={12}>
                    <FormGroup
                      fieldId="temperature"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Temperature: {temperatureSlider.toFixed(2)}</FlexItem>
                          <FlexItem>
                            <Tooltip content="Controls randomness in outputs. Lower values (0.0) make responses more focused and deterministic, higher values (2.0) make them more creative and random.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <Slider
                        id="temperature"
                        value={temperatureSlider}
                        onChange={(_, value) => handleTemperatureChange(value)}
                        min={0}
                        max={2}
                        step={0.1}
                        showTicks
                        customSteps={[
                          { value: 0, label: '0' },
                          { value: 0.5, label: '0.5' },
                          { value: 1, label: '1.0' },
                          { value: 1.5, label: '1.5' },
                          { value: 2, label: '2.0' },
                        ]}
                      />
                      <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} style={{ marginTop: '0.25rem', fontSize: '0.75rem', color: '#6a6e73' }}>
                        <FlexItem>Focused</FlexItem>
                        <FlexItem>Creative</FlexItem>
                      </Flex>
                    </FormGroup>
                  </GridItem>

                  {/* Top P Slider */}
                  <GridItem span={12}>
                    <FormGroup
                      fieldId="top-p"
                      label={
                        <Flex spaceItems={{ default: 'spaceItemsXs' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <FlexItem>Top P (Nucleus Sampling): {topPSlider.toFixed(2)}</FlexItem>
                          <FlexItem>
                            <Tooltip content="Only consider tokens with cumulative probability up to this value. Lower values (0.1) make output more focused, higher values (1.0) consider more options.">
                              <InfoCircleIcon style={{ color: 'var(--pf-v5-global--Color--200)', cursor: 'help' }} />
                            </Tooltip>
                          </FlexItem>
                        </Flex>
                      }
                    >
                      <Slider
                        id="top-p"
                        value={topPSlider}
                        onChange={(_, value) => handleTopPChange(value)}
                        min={0}
                        max={1}
                        step={0.05}
                        showTicks
                        customSteps={[
                          { value: 0, label: '0' },
                          { value: 0.25, label: '0.25' },
                          { value: 0.5, label: '0.5' },
                          { value: 0.75, label: '0.75' },
                          { value: 1, label: '1.0' },
                        ]}
                      />
                      <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} style={{ marginTop: '0.25rem', fontSize: '0.75rem', color: '#6a6e73' }}>
                        <FlexItem>Focused</FlexItem>
                        <FlexItem>Diverse</FlexItem>
                      </Flex>
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

              {/* Configure and Test Buttons */}
              <Flex style={{ marginTop: '1rem' }}>
                <FlexItem>
                  <Button
                    variant="primary"
                    onClick={handleConfigure}
                    isDisabled={!isFormValid()}
                    isLoading={isConfiguring}
                  >
                    {isConfigured ? 'Update Configuration' : 'Apply Configuration'}
                  </Button>
                </FlexItem>
                <FlexItem>
                  <Button
                    variant="secondary"
                    onClick={handleTestConnection}
                    isDisabled={!model || !apiBase}
                    isLoading={isTestingConnection}
                    icon={<PlayIcon />}
                  >
                    Test Connection
                  </Button>
                </FlexItem>
              </Flex>

              {/* Test Result Inline Display */}
              {testResult && !showTestModal && (
                <Alert
                  variant={testResult.success ? AlertVariant.success : AlertVariant.danger}
                  isInline
                  title={testResult.success ? 'Connection Successful' : 'Connection Failed'}
                  style={{ marginTop: '1rem' }}
                >
                  {testResult.success ? (
                    <div>
                      <p><strong>Response:</strong> {testResult.response}</p>
                      <p><strong>Latency:</strong> {testResult.latency}ms</p>
                    </div>
                  ) : (
                    <p>{testResult.error}</p>
                  )}
                </Alert>
              )}

              {(isConfigured || (modelConfig && modelConfig.model)) && (
                <Alert
                  variant={AlertVariant.success}
                  isInline
                  title="Configuration applied"
                  style={{ marginTop: '1rem' }}
                />
              )}

              {/* Test Connection Modal */}
              <Modal
                variant={ModalVariant.medium}
                title="Test Model Connection"
                isOpen={showTestModal}
                onClose={() => setShowTestModal(false)}
                actions={[
                  <Button key="close" variant="primary" onClick={() => setShowTestModal(false)}>
                    Close
                  </Button>
                ]}
              >
                <div style={{ padding: '1rem' }}>
                  <p style={{ marginBottom: '1rem' }}>
                    Testing connection with prompt: <em>"What is the capital of France? Answer in one word."</em>
                  </p>
                  
                  {isTestingConnection ? (
                    <div style={{ textAlign: 'center', padding: '2rem' }}>
                      <Spinner size="lg" />
                      <p style={{ marginTop: '1rem' }}>Sending test request...</p>
                    </div>
                  ) : testResult ? (
                    <Alert
                      variant={testResult.success ? AlertVariant.success : AlertVariant.danger}
                      title={testResult.success ? 'Connection Successful!' : 'Connection Failed'}
                    >
                      {testResult.success ? (
                        <div>
                          <p style={{ marginBottom: '0.5rem' }}><strong>Model Response:</strong></p>
                          <CodeBlock>
                            <CodeBlockCode>{testResult.response}</CodeBlockCode>
                          </CodeBlock>
                          <p style={{ marginTop: '1rem' }}><strong>Latency:</strong> {testResult.latency}ms</p>
                        </div>
                      ) : (
                        <div>
                          <p><strong>Error:</strong> {testResult.error}</p>
                          <p style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                            Please check your model name, API base URL, and API key.
                          </p>
                        </div>
                      )}
                    </Alert>
                  ) : null}
                </div>
              </Modal>
            </Form>
          </CardBody>
        </Card>
      </GridItem>

      {/* Right Panel - Recommendations from Selected Flow */}
      <GridItem span={5} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Recommendations for {selectedFlow?.name || 'Selected Flow'}
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            {/* Show recommendations from the selected flow */}
            {selectedFlow?.recommended_models ? (
              <>
                {/* Hint about clicking model names */}
                <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: '#f0f0f0', borderRadius: '4px', fontSize: '0.85rem' }}>
                  💡 <strong>Tip:</strong> Click any model name below to use it. Make sure to verify the <strong>inference provider prefix</strong> (e.g., <code>hosted_vllm/</code>), the <strong>API Base URL</strong>, and the <strong>API Key</strong> match your setup.
                </div>
                
                <DescriptionList isHorizontal>
                <DescriptionListGroup>
                  <DescriptionListTerm>Default Model</DescriptionListTerm>
                  <DescriptionListDescription>
                    {selectedFlow.recommended_models.default ? (
                      <Button
                        variant="link"
                        isInline
                        onClick={() => {
                          setModel(`hosted_vllm/${selectedFlow.recommended_models.default}`);
                          setApiBase('http://localhost:8001/v1');
                        }}
                        style={{ padding: 0 }}
                      >
                        <code>{selectedFlow.recommended_models.default}</code>
                      </Button>
                    ) : (
                      <span style={{ color: '#6a6e73' }}>N/A</span>
                    )}
                  </DescriptionListDescription>
                </DescriptionListGroup>

                {selectedFlow.recommended_models.compatible?.length > 0 && (
                  <DescriptionListGroup>
                    <DescriptionListTerm>Compatible Models</DescriptionListTerm>
                    <DescriptionListDescription>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {selectedFlow.recommended_models.compatible.map((modelName) => (
                          <Button
                            key={modelName}
                            variant="link"
                            isInline
                            onClick={() => {
                              setModel(`hosted_vllm/${modelName}`);
                              setApiBase('http://localhost:8001/v1');
                            }}
                            style={{ padding: 0, textAlign: 'left' }}
                          >
                            <code>{modelName}</code>
                          </Button>
                        ))}
                      </div>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                )}

                {selectedFlow.recommended_models.experimental?.length > 0 && (
                  <DescriptionListGroup>
                    <DescriptionListTerm>Experimental</DescriptionListTerm>
                    <DescriptionListDescription>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {selectedFlow.recommended_models.experimental.map((modelName) => (
                          <Button
                            key={modelName}
                            variant="link"
                            isInline
                            onClick={() => {
                              setModel(`hosted_vllm/${modelName}`);
                              setApiBase('http://localhost:8001/v1');
                            }}
                            style={{ padding: 0, textAlign: 'left', color: '#6a6e73' }}
                          >
                            <code>{modelName}</code>
                            <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>(experimental)</span>
                          </Button>
                        ))}
                      </div>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                )}
              </DescriptionList>
              </>
            ) : (
              <Alert variant={AlertVariant.info} isInline title="No model recommendations">
                This flow does not have specific model recommendations. You can use any compatible LLM.
              </Alert>
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

