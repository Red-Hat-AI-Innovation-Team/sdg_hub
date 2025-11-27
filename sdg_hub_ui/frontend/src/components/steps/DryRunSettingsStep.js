import React, { useState } from 'react';
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
  ExpandableSection,
} from '@patternfly/react-core';
import { PlayIcon, CheckCircleIcon, ExclamationCircleIcon } from '@patternfly/react-icons';
import { savedConfigAPI } from '../../services/api';
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
  datasetConfig 
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [dryRunResult, setDryRunResult] = useState(null);
  const [dryRunError, setDryRunError] = useState(null);
  const [dryRunOutput, setDryRunOutput] = useState('');
  const [outputExpanded, setOutputExpanded] = useState(false);

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
    let eventSource = null;

    try {
      // Step 1: Create a temporary config for the dry run
      const tempConfig = {
        flow_name: selectedFlow.name,
        flow_id: selectedFlow.id,
        flow_path: selectedFlow.path,
        model_configuration: modelConfig,
        dataset_configuration: datasetConfig,
        dry_run_configuration: dryRunConfig,
        tags: selectedFlow.tags || [],
        status: 'configured',
      };

      // Save temp config
      const saveResponse = await savedConfigAPI.save(tempConfig);
      const configId = saveResponse.configuration?.id;

      setDryRunOutput(prev => prev + '✅ Configuration prepared\n📊 Loading dataset...\n');

      // Step 2: Load dataset
      if (datasetConfig && datasetConfig.data_files && datasetConfig.data_files !== '.') {
        await axios.post('http://localhost:8000/api/dataset/load', datasetConfig);
        setDryRunOutput(prev => prev + '✅ Dataset loaded\n🚀 Starting dry run...\n\n');
      }

      // Step 3: Run dry run with streaming
      const params = new URLSearchParams({
        sample_size: dryRunConfig?.sample_size || 2,
        enable_time_estimation: dryRunConfig?.enable_time_estimation || true,
        max_concurrency: dryRunConfig?.max_concurrency || 10,
      });

      const url = `http://localhost:8000/api/flow/dry-run-stream?${params}`;
      eventSource = new EventSource(url);

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'start' || data.type === 'log') {
            setDryRunOutput(prev => prev + data.message + '\n');
          } else if (data.type === 'complete') {
            setDryRunOutput(prev => prev + `\n✅ Dry run completed in ${data.result?.execution_time_seconds?.toFixed(2)}s\n`);
            setDryRunResult(data.result);
            setIsRunning(false);
            eventSource.close();
          } else if (data.type === 'error') {
            setDryRunOutput(prev => prev + `\n❌ Error: ${data.message}\n`);
            setDryRunError(data.message);
            setIsRunning(false);
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
        eventSource.close();
      };

    } catch (error) {
      console.error('Dry run error:', error);
      setDryRunOutput(prev => prev + `\n❌ Error: ${error.message}\n`);
      setDryRunError(error.response?.data?.detail || error.message);
      setIsRunning(false);
      if (eventSource) eventSource.close();
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
                {/* Sample Size */}
                <FormGroup 
                  label="Sample Size" 
                  isRequired 
                  fieldId="sample-size"
                  helperText="Number of samples to use for dry run testing (1-10 recommended)"
                >
                  <NumberInput
                    id="sample-size"
                    value={dryRunConfig?.sample_size || 2}
                    onMinus={() => handleChange('sample_size', Math.max(1, (dryRunConfig?.sample_size || 2) - 1))}
                    onPlus={() => handleChange('sample_size', Math.min(10, (dryRunConfig?.sample_size || 2) + 1))}
                    onChange={(event) => {
                      const value = parseInt(event.target.value, 10);
                      if (!isNaN(value) && value >= 1 && value <= 10) {
                        handleChange('sample_size', value);
                      }
                    }}
                    min={1}
                    max={10}
                    widthChars={4}
                  />
                </FormGroup>

                {/* Enable Time Estimation */}
                <FormGroup fieldId="time-estimation">
                  <Checkbox
                    id="time-estimation"
                    label="Enable time estimation"
                    description="Estimate total execution time for the full dataset"
                    isChecked={dryRunConfig?.enable_time_estimation !== false}
                    onChange={(event, checked) => handleChange('enable_time_estimation', checked)}
                  />
                </FormGroup>

                {/* Max Concurrency */}
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

                {/* Run Dry Run Button */}
                <div style={{ marginTop: '24px' }}>
                  <Button
                    variant="primary"
                    icon={isRunning ? <Spinner size="sm" /> : <PlayIcon />}
                    onClick={handleRunDryRun}
                    isDisabled={isRunning || !canRunDryRun()}
                    isLoading={isRunning}
                  >
                    {isRunning ? 'Running Dry Run...' : 'Run Dry Run'}
                  </Button>
                  
                  {!canRunDryRun() && (
                    <p style={{ marginTop: '8px', fontSize: '12px', color: '#f0ab00' }}>
                      Complete flow, model, and dataset configuration first
                    </p>
                  )}
                </div>
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
                <ExpandableSection
                  toggleText={outputExpanded ? 'Hide output logs' : 'Show output logs'}
                  onToggle={() => setOutputExpanded(!outputExpanded)}
                  isExpanded={outputExpanded}
                >
                  <CodeBlock>
                    <CodeBlockCode style={{ 
                      maxHeight: '300px', 
                      overflow: 'auto',
                      fontSize: '12px',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {dryRunOutput}
                    </CodeBlockCode>
                  </CodeBlock>
                </ExpandableSection>
              )}
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
    </div>
  );
};

export default DryRunSettingsStep;
