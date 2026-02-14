import React, { useState, useRef, useEffect } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Form,
  FormGroup,
  NumberInput,
  Checkbox,
  Alert,
  AlertVariant,
  Button,
  Spinner,
  Grid,
  GridItem,
  CodeBlock,
  CodeBlockCode,
} from '@patternfly/react-core';
import { PlayIcon, CheckCircleIcon, ExclamationCircleIcon, StopIcon } from '@patternfly/react-icons';
import axios from 'axios';

/**
 * Dry Run Step
 * Configure and run dry run directly from the wizard
 */
const DryRunSettingsStep = ({ 
  dryRunConfig, 
  onConfigChange, 
  selectedFlow,
  modelConfig,
  datasetConfig,
  onDryRunStateChange,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [dryRunResult, setDryRunResult] = useState(null);
  const [dryRunError, setDryRunError] = useState(null);
  const [dryRunOutput, setDryRunOutput] = useState('');
  const eventSourceRef = useRef(null);
  const outputEndRef = useRef(null);

  // Check if this is a small dataset (5 or fewer samples)
  const isSmallDataset = datasetConfig?.num_samples && datasetConfig.num_samples <= 5;

  // For small datasets, force sample_size=1, max_concurrency=1, and disable time estimation
  useEffect(() => {
    if (isSmallDataset) {
      const updates = {};
      if (dryRunConfig?.sample_size !== 1) updates.sample_size = 1;
      if (dryRunConfig?.max_concurrency !== 1) updates.max_concurrency = 1;
      if (dryRunConfig?.enable_time_estimation !== false) updates.enable_time_estimation = false;
      
      if (Object.keys(updates).length > 0) {
        onConfigChange({
          ...dryRunConfig,
          ...updates,
        });
      }
    }
  }, [isSmallDataset, dryRunConfig, onConfigChange]);

  // Notify parent of dry run state changes
  useEffect(() => {
    if (onDryRunStateChange) {
      onDryRunStateChange(isRunning);
    }
  }, [isRunning, onDryRunStateChange]);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  // Auto-scroll to bottom when output changes
  useEffect(() => {
    if (outputEndRef.current) {
      outputEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [dryRunOutput]);

  const handleChange = (field, value) => {
    onConfigChange({
      ...dryRunConfig,
      [field]: value
    });
  };

  /**
   * Check if configuration is complete enough for dry run
   */
  const canRunDryRun = () => {
    return selectedFlow && modelConfig?.model && datasetConfig?.data_files;
  };

  /**
   * Run the dry run
   */
  const handleRunDryRun = async () => {
    if (!canRunDryRun()) {
      setDryRunError('Please complete the flow, model, and dataset configuration before running a dry run.');
      return;
    }

    setIsRunning(true);
    setDryRunResult(null);
    setDryRunError(null);
    setDryRunOutput('🔧 Preparing dry run...\n');

    try {
      // Step 1: Select the flow on the backend (use flow name endpoint)
      setDryRunOutput(prev => prev + '📋 Loading flow...\n');
      await axios.post(`http://localhost:8000/api/flows/${encodeURIComponent(selectedFlow.name)}/select`);
      setDryRunOutput(prev => prev + '✅ Flow loaded\n');

      // Step 2: Apply model configuration
      setDryRunOutput(prev => prev + '🔧 Applying model configuration...\n');
      await axios.post('http://localhost:8000/api/model/configure', modelConfig);
      setDryRunOutput(prev => prev + '✅ Model configured\n');

      // Step 3: Load dataset
      setDryRunOutput(prev => prev + '📊 Loading dataset...\n');
      if (datasetConfig && datasetConfig.data_files && datasetConfig.data_files !== '.') {
        await axios.post('http://localhost:8000/api/dataset/load', datasetConfig);
        setDryRunOutput(prev => prev + '✅ Dataset loaded\n🚀 Starting dry run...\n\n');
      }

      // Step 4: Run dry run with streaming (no config_id needed, uses current backend state)
      const params = new URLSearchParams({
        sample_size: dryRunConfig?.sample_size || 2,
        enable_time_estimation: dryRunConfig?.enable_time_estimation || true,
        max_concurrency: dryRunConfig?.max_concurrency || 10,
      });

      const url = `http://localhost:8000/api/flow/dry-run-stream?${params}`;
      
      // Close any existing EventSource
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'start' || data.type === 'log') {
            setDryRunOutput(prev => prev + data.message + '\n');
          } else if (data.type === 'complete') {
            setDryRunOutput(prev => prev + `\n✅ Dry run completed in ${data.result?.execution_time_seconds?.toFixed(2)}s\n`);
            setDryRunResult(data.result);
            setIsRunning(false);
            setIsCancelling(false);
            eventSourceRef.current = null;
            eventSource.close();
          } else if (data.type === 'cancelled') {
            setDryRunOutput(prev => prev + `\n⚠️ ${data.message}\n`);
            setIsRunning(false);
            setIsCancelling(false);
            eventSourceRef.current = null;
            eventSource.close();
          } else if (data.type === 'error') {
            setDryRunOutput(prev => prev + `\n❌ Error: ${data.message}\n`);
            setDryRunError(data.message);
            setIsRunning(false);
            setIsCancelling(false);
            eventSourceRef.current = null;
            eventSource.close();
          }
        } catch (err) {
          console.error('Error parsing event:', err);
        }
      };

      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        setDryRunOutput(prev => prev + '\n❌ Connection to server lost\n');
        setDryRunError('Connection to server lost');
        setIsRunning(false);
        setIsCancelling(false);
        eventSourceRef.current = null;
        eventSource.close();
      };

    } catch (error) {
      console.error('Dry run error:', error);
      setDryRunOutput(prev => prev + `\n❌ Error: ${error.message}\n`);
      setDryRunError(error.response?.data?.detail || error.message);
      setIsRunning(false);
      setIsCancelling(false);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    }
  };

  /**
   * Stop the dry run
   */
  const handleStopDryRun = async () => {
    try {
      setIsCancelling(true);
      
      // Close the EventSource first
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      
      // Call backend to cancel
      await axios.post('http://localhost:8000/api/flow/cancel-dry-run');
      
      setDryRunOutput(prev => prev + '\n⚠️ Dry run stopped by user\n');
      setIsRunning(false);
      setIsCancelling(false);
    } catch (error) {
      console.error('Error stopping dry run:', error);
      setIsCancelling(false);
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '1.5rem 2.5rem' }}>
      <Alert
        variant={AlertVariant.info}
        isInline
        title="Test your configuration"
        style={{ marginBottom: '24px', flexShrink: 0 }}
      >
        Configure dry run settings and test your configuration before proceeding. This helps validate your setup and estimate execution time.
      </Alert>

      <Grid hasGutter style={{ flex: 1 }}>
        {/* Left side - Settings */}
        <GridItem span={6}>
          <Card isFullHeight>
            <CardTitle>
              <Title headingLevel="h2" size="xl">
                Dry Run Settings
              </Title>
            </CardTitle>
            <CardBody>
              <Form>
                {/* Small Dataset Notice */}
                {isSmallDataset && (
                  <Alert
                    variant={AlertVariant.info}
                    isInline
                    title="Small dataset detected"
                    style={{ marginBottom: '16px' }}
                  >
                    Your dataset has {datasetConfig.num_samples} sample{datasetConfig.num_samples !== 1 ? 's' : ''}. 
                    Sample size is set to 1 and advanced options are hidden.
                  </Alert>
                )}

                {/* Sample Size */}
                <FormGroup 
                  label="Sample Size" 
                  isRequired 
                  fieldId="sample-size"
                  helperText={isSmallDataset 
                    ? "Fixed to 1 for small datasets (5 or fewer samples)" 
                    : "Number of samples to use for dry run testing (1-10 recommended)"
                  }
                >
                  <NumberInput
                    id="sample-size"
                    value={isSmallDataset ? 1 : (dryRunConfig?.sample_size || 2)}
                    onMinus={() => !isSmallDataset && handleChange('sample_size', Math.max(1, (dryRunConfig?.sample_size || 2) - 1))}
                    onPlus={() => !isSmallDataset && handleChange('sample_size', Math.min(10, (dryRunConfig?.sample_size || 2) + 1))}
                    onChange={(event) => {
                      if (isSmallDataset) return;
                      const value = parseInt(event.target.value, 10);
                      if (!isNaN(value) && value >= 1 && value <= 10) {
                        handleChange('sample_size', value);
                      }
                    }}
                    min={1}
                    max={10}
                    widthChars={4}
                    isDisabled={isSmallDataset}
                  />
                </FormGroup>

                {/* Enable Time Estimation - hidden for small datasets */}
                {!isSmallDataset && (
                  <FormGroup fieldId="time-estimation">
                    <Checkbox
                      id="time-estimation"
                      label="Enable time estimation"
                      description="Estimate total execution time for the full dataset"
                      isChecked={dryRunConfig?.enable_time_estimation !== false}
                      onChange={(event, checked) => handleChange('enable_time_estimation', checked)}
                    />
                  </FormGroup>
                )}

                {/* Max Concurrency - hidden for small datasets */}
                {!isSmallDataset && (
                  <FormGroup 
                    label="Max Concurrency" 
                    fieldId="max-concurrency"
                    helperText="Maximum number of concurrent LLM requests (1-200)"
                  >
                    <NumberInput
                      id="max-concurrency"
                      value={dryRunConfig?.max_concurrency || 10}
                      onMinus={() => handleChange('max_concurrency', Math.max(1, (dryRunConfig?.max_concurrency || 10) - 10))}
                      onPlus={() => handleChange('max_concurrency', Math.min(200, (dryRunConfig?.max_concurrency || 10) + 10))}
                      onChange={(event) => {
                        const value = parseInt(event.target.value, 10);
                        if (!isNaN(value) && value >= 1 && value <= 200) {
                          handleChange('max_concurrency', value);
                        }
                      }}
                      min={1}
                      max={200}
                      widthChars={6}
                    />
                  </FormGroup>
                )}

                {/* Run/Stop Dry Run Buttons */}
                <div style={{ marginTop: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                  {!isRunning ? (
                    <Button
                      variant="primary"
                      icon={<PlayIcon />}
                      onClick={handleRunDryRun}
                      isDisabled={!canRunDryRun()}
                    >
                      Run Dry Run
                    </Button>
                  ) : (
                    <Button
                      variant="danger"
                      icon={!isCancelling && <StopIcon />}
                      onClick={handleStopDryRun}
                      isLoading={isCancelling}
                    >
                      {isCancelling ? 'Stopping...' : 'Stop Dry Run'}
                    </Button>
                  )}
                  
                  {isRunning && (
                    <span style={{ color: '#0066cc', fontSize: '14px' }}>
                      <Spinner size="sm" style={{ marginRight: '8px' }} />
                      Dry run in progress...
                    </span>
                  )}
                </div>
                
                {!canRunDryRun() && !isRunning && (
                  <p style={{ marginTop: '8px', fontSize: '12px', color: '#f0ab00' }}>
                    Complete flow, model, and dataset configuration first
                  </p>
                )}
              </Form>
            </CardBody>
          </Card>
        </GridItem>

        {/* Right side - Results */}
        <GridItem span={6}>
          <Card isFullHeight>
            <CardTitle>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Title headingLevel="h2" size="xl">
                  Dry Run Results
                </Title>
                {dryRunResult && <CheckCircleIcon color="#3e8635" />}
                {dryRunError && <ExclamationCircleIcon color="#c9190b" />}
              </div>
            </CardTitle>
            <CardBody style={{ display: 'flex', flexDirection: 'column' }}>
              {!dryRunResult && !dryRunError && !dryRunOutput && (
                <div style={{ 
                  flex: 1, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: '#6a6e73'
                }}>
                  <p>Run a dry run to see results here</p>
                </div>
              )}

              {dryRunError && (
                <Alert
                  variant={AlertVariant.danger}
                  isInline
                  title="Dry run failed"
                  style={{ marginBottom: '16px' }}
                >
                  {dryRunError}
                </Alert>
              )}

              {dryRunResult && (
                <Alert
                  variant={AlertVariant.success}
                  isInline
                  title="Dry run successful!"
                  style={{ marginBottom: '16px' }}
                >
                  <div style={{ marginTop: '8px' }}>
                    <strong>Execution time:</strong> {dryRunResult.execution_time_seconds?.toFixed(2)}s
                    <br />
                    <strong>Samples processed:</strong> {dryRunResult.num_samples || dryRunConfig?.sample_size || 2}
                  </div>
                </Alert>
              )}

              {dryRunOutput && (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                  <Title headingLevel="h4" size="md" style={{ marginBottom: '8px', flexShrink: 0 }}>
                    Output Logs
                  </Title>
                  <CodeBlock style={{ flex: 1, minHeight: 0 }}>
                    <CodeBlockCode style={{ 
                      height: '300px', 
                      overflow: 'auto',
                      fontSize: '12px',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace',
                    }}>
                      {dryRunOutput}
                      <div ref={outputEndRef} />
                    </CodeBlockCode>
                  </CodeBlock>
                </div>
              )}
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
    </div>
  );
};

export default DryRunSettingsStep;
