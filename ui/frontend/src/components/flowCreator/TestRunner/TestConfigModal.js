import React, { useState, useEffect, useMemo } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Form,
  FormGroup,
  TextInput,
  TextArea,
  Alert,
  AlertVariant,
  Spinner,
  FormHelperText,
  HelperText,
  HelperTextItem,
  Title,
  Divider,
  Radio,
  Card,
  CardBody,
} from '@patternfly/react-core';
import { PlayIcon, CogIcon, FileIcon, EditIcon } from '@patternfly/react-icons';

/**
 * Pre-configured sample data for common template column patterns
 */
const SAMPLE_DATA_TEMPLATES = {
  // Sample document for testing
  document: `Artificial Intelligence (AI) has revolutionized many industries in recent years. Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning, which uses neural networks with many layers, has achieved remarkable results in image recognition, natural language processing, and game playing.

The development of large language models (LLMs) has been particularly transformative. These models, trained on vast amounts of text data, can generate human-like text, answer questions, summarize documents, and assist with coding tasks. Companies like OpenAI, Google, and Anthropic have developed increasingly capable models.

However, AI also raises important ethical considerations. Issues such as bias in training data, job displacement, and the potential for misuse must be carefully addressed. Researchers and policymakers are working together to develop guidelines and regulations to ensure AI is developed and deployed responsibly.`,

  // Sample document outline
  document_outline: `1. Introduction to AI
   - Definition and scope
   - Historical context
2. Machine Learning Fundamentals
   - Supervised learning
   - Unsupervised learning
   - Reinforcement learning
3. Deep Learning
   - Neural network architectures
   - Applications and breakthroughs
4. Large Language Models
   - Training methodology
   - Capabilities and limitations
5. Ethical Considerations
   - Bias and fairness
   - Societal impact`,

  // Sample domain
  domain: 'Technology / Artificial Intelligence',

  // Sample ICL (In-Context Learning) document
  icl_document: `Cloud computing has transformed how businesses operate by providing on-demand access to computing resources over the internet. Instead of maintaining physical servers, organizations can rent computing power, storage, and applications from cloud providers like Amazon Web Services, Microsoft Azure, and Google Cloud Platform.

The three main service models are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Each offers different levels of control and responsibility, allowing businesses to choose the model that best fits their needs.`,

  // Sample ICL Q&A pairs
  icl_query_1: 'What are the main benefits of cloud computing for businesses?',
  icl_response_1: 'The main benefits include: 1) Cost savings through pay-as-you-go pricing, 2) Scalability to handle varying workloads, 3) Accessibility from anywhere with internet, 4) Reduced IT maintenance burden, and 5) Faster deployment of applications and services.',

  icl_query_2: 'How do IaaS, PaaS, and SaaS differ from each other?',
  icl_response_2: 'IaaS provides basic infrastructure (servers, storage, networking) with maximum control. PaaS adds development tools and runtime environments, simplifying application development. SaaS delivers complete applications ready to use, requiring no technical management from the user.',

  icl_query_3: 'What security considerations should organizations keep in mind with cloud computing?',
  icl_response_3: 'Organizations should consider: data encryption (at rest and in transit), access control and identity management, compliance with industry regulations, shared responsibility model understanding, regular security audits, and disaster recovery planning.',

  // For text analysis flows
  text: `The rise of remote work has fundamentally changed the modern workplace. Since 2020, millions of employees have transitioned to working from home, leading to significant shifts in work culture, communication patterns, and productivity metrics. Companies have invested heavily in collaboration tools, virtual meeting platforms, and cybersecurity measures to support distributed teams. While remote work offers flexibility and eliminates commuting, it also presents challenges such as maintaining team cohesion, preventing burnout, and ensuring work-life balance.`,
};

/**
 * Get sample data based on required columns
 */
const getSampleDataForColumns = (requiredColumns) => {
  const sampleData = {};
  requiredColumns.forEach(col => {
    if (SAMPLE_DATA_TEMPLATES[col]) {
      sampleData[col] = SAMPLE_DATA_TEMPLATES[col];
    } else {
      // Generate placeholder for unknown columns
      sampleData[col] = `Sample ${col} content for testing`;
    }
  });
  return sampleData;
};

/**
 * Test Configuration Modal
 * 
 * Allows users to configure a test model and provide sample input
 * to test their flow before saving.
 */
const TestConfigModal = ({ 
  isOpen, 
  onClose, 
  onRunTest, 
  requiredColumns = [],
  isRunning = false,
}) => {
  const [modelName, setModelName] = useState('');
  const [apiBase, setApiBase] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [sampleData, setSampleData] = useState({});
  const [errors, setErrors] = useState({});
  const [dataMode, setDataMode] = useState('example'); // 'example' or 'custom'

  // Get pre-configured sample data for the required columns
  const exampleData = useMemo(() => {
    return getSampleDataForColumns(requiredColumns);
  }, [requiredColumns]);

  // Check if we have example data available
  const hasExampleData = useMemo(() => {
    return requiredColumns.length > 0 && requiredColumns.some(col => SAMPLE_DATA_TEMPLATES[col]);
  }, [requiredColumns]);

  /**
   * Initialize sample data fields from required columns
   */
  useEffect(() => {
    if (dataMode === 'example' && hasExampleData) {
      setSampleData(exampleData);
    } else {
      // Initialize with empty strings for custom mode
      const initialData = {};
      requiredColumns.forEach(col => {
        initialData[col] = sampleData[col] || '';
      });
      setSampleData(initialData);
    }
  }, [requiredColumns, dataMode, hasExampleData, exampleData]);

  /**
   * Handle data mode change
   */
  const handleDataModeChange = (mode) => {
    setDataMode(mode);
    if (mode === 'example') {
      setSampleData(exampleData);
    } else {
      // Keep current values or initialize empty
      const customData = {};
      requiredColumns.forEach(col => {
        customData[col] = '';
      });
      setSampleData(customData);
    }
    setErrors({});
  };

  /**
   * Handle sample data field change
   */
  const handleSampleDataChange = (field, value) => {
    setSampleData(prev => ({
      ...prev,
      [field]: value,
    }));
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  /**
   * Clear error for a field
   */
  const clearError = (field) => {
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  /**
   * Validate and run test
   */
  const handleRunTest = () => {
    const newErrors = {};

    // Validate model name
    if (!modelName.trim()) {
      newErrors.modelName = 'Model name is required';
    }

    // Validate required sample data
    requiredColumns.forEach(col => {
      if (!sampleData[col]?.trim()) {
        newErrors[col] = `${col} is required`;
      }
    });

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    // Run the test with proper model config structure
    onRunTest({
      modelConfig: {
        model: modelName,
        api_base: apiBase || undefined,
        api_key: apiKey || undefined,
      },
      sampleData,
    });
  };

  return (
    <Modal
      variant={ModalVariant.large}
      title="Test Flow Configuration"
      isOpen={isOpen}
      onClose={onClose}
      actions={[
        <Button
          key="run"
          variant="primary"
          icon={isRunning ? <Spinner size="sm" /> : <PlayIcon />}
          onClick={handleRunTest}
          isDisabled={isRunning}
        >
          {isRunning ? 'Running Test...' : 'Run Test'}
        </Button>,
        <Button key="cancel" variant="link" onClick={onClose} isDisabled={isRunning}>
          Cancel
        </Button>,
      ]}
    >
      <Form>
        {/* Model Configuration Section */}
        <Title headingLevel="h4" size="md" style={{ marginBottom: '16px' }}>
          <CogIcon style={{ marginRight: '8px' }} />
          Model Configuration
        </Title>

        <FormGroup label="Model Name" isRequired fieldId="model-name">
          <TextInput
            id="model-name"
            value={modelName}
            onChange={(event, value) => {
              setModelName(value);
              clearError('modelName');
            }}
            placeholder="e.g., gpt-4o-mini, claude-3-5-sonnet-20241022, hosted_vllm/llama-3.1-8b"
            validated={errors.modelName ? 'error' : 'default'}
          />
          <FormHelperText>
            <HelperText>
              <HelperTextItem>
                Enter the model name as expected by LiteLLM (e.g., gpt-4o-mini, anthropic/claude-3-5-sonnet, hosted_vllm/model-name)
              </HelperTextItem>
            </HelperText>
          </FormHelperText>
          {errors.modelName && (
            <FormHelperText>
              <HelperText>
                <HelperTextItem variant="error">{errors.modelName}</HelperTextItem>
              </HelperText>
            </FormHelperText>
          )}
        </FormGroup>

        <FormGroup label="Base URL" fieldId="api-base">
          <TextInput
            id="api-base"
            value={apiBase}
            onChange={(event, value) => {
              setApiBase(value);
              clearError('apiBase');
            }}
            placeholder="e.g., https://api.openai.com/v1 or http://localhost:8000/v1"
            validated={errors.apiBase ? 'error' : 'default'}
          />
          <FormHelperText>
            <HelperText>
              <HelperTextItem>
                Optional. Required for self-hosted models (vLLM, Ollama, etc.) or custom endpoints.
              </HelperTextItem>
            </HelperText>
          </FormHelperText>
        </FormGroup>

        <FormGroup label="API Key" fieldId="api-key">
          <TextInput
            id="api-key"
            type="password"
            value={apiKey}
            onChange={(event, value) => {
              setApiKey(value);
              clearError('apiKey');
            }}
            placeholder="Enter your API key (optional for local models)"
          />
        </FormGroup>

        <Divider style={{ margin: '24px 0' }} />

        {/* Sample Data Section */}
        <Title headingLevel="h4" size="md" style={{ marginBottom: '16px' }}>
          Sample Input Data
        </Title>

        {requiredColumns.length === 0 ? (
          <Alert
            variant={AlertVariant.info}
            isInline
            title="No input columns detected"
          >
            Configure your flow nodes to define required input columns.
          </Alert>
        ) : (
          <>
            {/* Data Mode Selection */}
            {hasExampleData && (
              <div style={{ marginBottom: '20px' }}>
                <Card isFlat isCompact style={{ marginBottom: '12px' }}>
                  <CardBody style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'flex', gap: '24px' }}>
                      <Radio
                        isChecked={dataMode === 'example'}
                        name="data-mode"
                        onChange={() => handleDataModeChange('example')}
                        label={
                          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <FileIcon />
                            <strong>Use Example Data</strong>
                            <span style={{ color: '#6a6e73', fontWeight: 'normal' }}>
                              - Pre-configured sample data for quick testing
                            </span>
                          </span>
                        }
                        id="data-mode-example"
                      />
                    </div>
                    <div style={{ display: 'flex', gap: '24px', marginTop: '12px' }}>
                      <Radio
                        isChecked={dataMode === 'custom'}
                        name="data-mode"
                        onChange={() => handleDataModeChange('custom')}
                        label={
                          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <EditIcon />
                            <strong>Configure Custom Data</strong>
                            <span style={{ color: '#6a6e73', fontWeight: 'normal' }}>
                              - Enter your own test data
                            </span>
                          </span>
                        }
                        id="data-mode-custom"
                      />
                    </div>
                  </CardBody>
                </Card>

                {dataMode === 'example' && (
                  <Alert
                    variant={AlertVariant.success}
                    isInline
                    title="Example data loaded"
                    style={{ marginBottom: '16px' }}
                  >
                    Pre-configured sample data has been loaded for {requiredColumns.length} required column(s). 
                    You can review and modify the data below, or switch to custom data to start fresh.
                  </Alert>
                )}
              </div>
            )}

            {!hasExampleData && (
              <Alert
                variant={AlertVariant.info}
                isInline
                title="Provide sample data"
                style={{ marginBottom: '16px' }}
              >
                Enter sample values for each input column to test your flow.
              </Alert>
            )}

            {/* Sample Data Fields */}
            {requiredColumns.map(column => (
              <FormGroup
                key={column}
                label={column}
                isRequired
                fieldId={`sample-${column}`}
              >
                <TextArea
                  id={`sample-${column}`}
                  value={sampleData[column] || ''}
                  onChange={(event, value) => handleSampleDataChange(column, value)}
                  placeholder={`Enter sample ${column}...`}
                  rows={
                    column === 'document' || column === 'icl_document' ? 8 :
                    column === 'document_outline' ? 6 :
                    column.includes('response') ? 4 :
                    column.includes('text') ? 6 : 3
                  }
                  validated={errors[column] ? 'error' : 'default'}
                  style={{ fontFamily: 'monospace', fontSize: '13px' }}
                />
                {errors[column] && (
                  <FormHelperText>
                    <HelperText>
                      <HelperTextItem variant="error">{errors[column]}</HelperTextItem>
                    </HelperText>
                  </FormHelperText>
                )}
              </FormGroup>
            ))}
          </>
        )}
      </Form>
    </Modal>
  );
};

export default TestConfigModal;
