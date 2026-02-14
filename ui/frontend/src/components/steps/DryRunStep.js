import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Form,
  FormGroup,
  NumberInput,
  Checkbox,
  Button,
  Alert,
  AlertVariant,
  Grid,
  GridItem,
  Spinner,
  Progress,
  ProgressMeasureLocation,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  ExpandableSection,
  CodeBlock,
  CodeBlockCode,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
} from '@patternfly/react-core';
import { PlayIcon, CheckCircleIcon, StopIcon } from '@patternfly/react-icons';
import { executionAPI } from '../../services/api';
import Convert from 'ansi-to-html';
import axios from 'axios';

/**
 * Dry Run Step Component
 * 
 * Allows users to:
 * - Configure dry run parameters (sample_size, time estimation, concurrency)
 * - Execute a dry run to test the configuration
 * - View execution results and time estimates
 * - See block-by-block execution details
 */
const DryRunStep = ({ onError, onDryRunStateChange }) => {
  const [dryRunResult, setDryRunResult] = useState(null);
  const [isDryRunning, setIsDryRunning] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  
  // Form state
  const [sampleSize, setSampleSize] = useState(2);
  const [enableTimeEstimation, setEnableTimeEstimation] = useState(true);
  const [maxConcurrency, setMaxConcurrency] = useState(100);
  const [useMaxConcurrency, setUseMaxConcurrency] = useState(true);
  
  // Real-time logs state
  const [executionLogs, setExecutionLogs] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // UI state
  const [isResultsExpanded, setIsResultsExpanded] = useState(true);
  const [isLogsExpanded, setIsLogsExpanded] = useState(true);
  const logsEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  
  // Single Convert instance with escapeXML enabled to prevent XSS
  const convertRef = useRef(new Convert({ fg: '#d4d4d4', bg: '#1e1e1e', escapeXML: true }));

  /**
   * Notify parent of dry run state changes
   */
  useEffect(() => {
    if (onDryRunStateChange) {
      onDryRunStateChange(isDryRunning);
    }
  }, [isDryRunning, onDryRunStateChange]);

  /**
   * Cleanup EventSource on unmount
   */
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  /**
   * Auto-scroll logs to bottom when new logs arrive
   */
  useEffect(() => {
    if (logsEndRef.current && isDryRunning) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [executionLogs, isDryRunning]);

  /**
   * Handle dry run execution with real-time log streaming
   */
  const handleRunDryRun = async () => {
    try {
      setIsDryRunning(true);
      setIsStreaming(true);
      setExecutionLogs([]);
      setDryRunResult(null);
      setIsLogsExpanded(true);
      
      // Verify backend has current configuration
      const currentConfig = await axios.get('http://localhost:8000/api/config/current');
      if (!currentConfig.data.flow || !currentConfig.data.model_config?.model || !currentConfig.data.dataset_info) {
        throw new Error('Configuration incomplete. Please complete all steps first.');
      }
      
      // Build URL with query parameters
      const params = new URLSearchParams();
      params.append('sample_size', String(sampleSize));
      params.append('enable_time_estimation', String(enableTimeEstimation));
      if (useMaxConcurrency) {
        params.append('max_concurrency', String(maxConcurrency));
      }
      
      const url = `http://localhost:8000/api/flow/dry-run-stream?${params}`;
      
      // Close any existing EventSource
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      
      // Create EventSource for server-sent events
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'start') {
            setExecutionLogs(prev => [...prev, { type: 'info', message: data.message }]);
          } else if (data.type === 'log') {
            setExecutionLogs(prev => [...prev, data]);
          } else if (data.type === 'complete') {
            setDryRunResult(data.result);
            setExecutionLogs(prev => [...prev, { 
              type: 'success', 
              message: `✅ Dry run completed in ${data.result.execution_time_seconds?.toFixed(2)}s` 
            }]);
            setIsDryRunning(false);
            setIsStreaming(false);
            setIsCancelling(false);
            if (eventSourceRef.current) {
              eventSourceRef.current.close();
              eventSourceRef.current = null;
            }
          } else if (data.type === 'cancelled') {
            setExecutionLogs(prev => [...prev, { 
              type: 'warning', 
              message: `⚠️ ${data.message}` 
            }]);
            setIsDryRunning(false);
            setIsStreaming(false);
            setIsCancelling(false);
            if (eventSourceRef.current) {
              eventSourceRef.current.close();
              eventSourceRef.current = null;
            }
          } else if (data.type === 'error') {
            onError('Dry run failed: ' + data.message);
            setIsDryRunning(false);
            setIsStreaming(false);
            setIsCancelling(false);
            if (eventSourceRef.current) {
              eventSourceRef.current.close();
              eventSourceRef.current = null;
            }
          }
        } catch (err) {
          console.error('Error parsing event:', err);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        onError('Connection to server lost');
        setIsDryRunning(false);
        setIsStreaming(false);
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
      };

    } catch (error) {
      // Clean up EventSource on error
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      onError('Dry run failed: ' + error.message);
      setIsDryRunning(false);
      setIsStreaming(false);
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
      
      setExecutionLogs(prev => [...prev, { 
        type: 'warning', 
        message: '⚠️ Dry run stopped by user' 
      }]);
      setIsDryRunning(false);
      setIsStreaming(false);
      setIsCancelling(false);
    } catch (error) {
      console.error('Error stopping dry run:', error);
      setIsCancelling(false);
    }
  };

  /**
   * Reset dry run results
   */
  const handleReset = () => {
    setDryRunResult(null);
    setExecutionLogs([]);
    setIsResultsExpanded(true);
    setIsLogsExpanded(true);
  };

  return (
    <Grid hasGutter>
      {/* Configuration Panel */}
      <GridItem span={12}>
        <Card>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Dry Run Configuration
            </Title>
          </CardTitle>
          <CardBody>
            <p style={{ marginBottom: '1.5rem' }}>
              Test your flow configuration with a small sample before running the full generation. 
              This helps validate settings and estimate execution time.
            </p>
            
            <Form>
              <Grid hasGutter>
                <GridItem span={6}>
                  {/* Sample Size */}
                  <FormGroup 
                    label="Sample Size" 
                    isRequired 
                    fieldId="sample-size"
                    helperText="Number of samples to use for testing (1-10 recommended)"
                  >
                    <NumberInput
                      id="sample-size"
                      value={sampleSize}
                      onMinus={() => setSampleSize(Math.max(1, sampleSize - 1))}
                      onPlus={() => setSampleSize(Math.min(10, sampleSize + 1))}
                      onChange={(event) => {
                        const value = parseInt(event.target.value, 10);
                        if (!isNaN(value) && value >= 1 && value <= 10) {
                          setSampleSize(value);
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
                      isChecked={enableTimeEstimation}
                      onChange={(event, checked) => setEnableTimeEstimation(checked)}
                    />
                  </FormGroup>
                </GridItem>

                <GridItem span={6}>
                  {/* Max Concurrency */}
                  <FormGroup fieldId="use-concurrency">
                    <Checkbox
                      id="use-concurrency"
                      label="Limit concurrent requests"
                      description="Control the number of parallel LLM requests"
                      isChecked={useMaxConcurrency}
                      onChange={(event, checked) => setUseMaxConcurrency(checked)}
                    />
                  </FormGroup>

                  {useMaxConcurrency && (
                    <FormGroup 
                      label="Max Concurrency" 
                      fieldId="max-concurrency"
                      helperText="Maximum number of concurrent LLM requests (1-200)"
                    >
                      <NumberInput
                        id="max-concurrency"
                        value={maxConcurrency}
                        onMinus={() => setMaxConcurrency(Math.max(1, maxConcurrency - 10))}
                        onPlus={() => setMaxConcurrency(Math.min(200, maxConcurrency + 10))}
                        onChange={(event) => {
                          const value = parseInt(event.target.value, 10);
                          if (!isNaN(value) && value >= 1 && value <= 200) {
                            setMaxConcurrency(value);
                          }
                        }}
                        min={1}
                        max={200}
                        widthChars={6}
                      />
                    </FormGroup>
                  )}
                </GridItem>
              </Grid>

              {/* Action Buttons */}
              <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
                {!isDryRunning ? (
                  <Button
                    variant="primary"
                    onClick={handleRunDryRun}
                    icon={<PlayIcon />}
                  >
                    Run Dry Run
                  </Button>
                ) : (
                  <Button
                    variant="danger"
                    onClick={handleStopDryRun}
                    isLoading={isCancelling}
                    icon={!isCancelling && <StopIcon />}
                  >
                    {isCancelling ? 'Stopping...' : 'Stop Dry Run'}
                  </Button>
                )}
                {dryRunResult && !isDryRunning && (
                  <Button
                    variant="secondary"
                    onClick={handleReset}
                  >
                    Reset
                  </Button>
                )}
              </div>
            </Form>
          </CardBody>
        </Card>
      </GridItem>

      {/* Real-time Execution Logs */}
      {(isDryRunning || executionLogs.length > 0) && (
        <GridItem span={12}>
          <Card>
            <CardTitle>
              <Title headingLevel="h2" size="xl">
                {isDryRunning ? '⚡ Executing Dry Run...' : '📋 Execution Logs'}
              </Title>
            </CardTitle>
            <CardBody>
              <ExpandableSection
                toggleText="Real-Time Execution Logs (Terminal View)"
                isExpanded={isLogsExpanded}
                onToggle={() => setIsLogsExpanded(!isLogsExpanded)}
              >
                <div style={{
                  fontSize: '14px',
                  fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                  maxHeight: '600px',
                  overflowY: 'auto',
                  backgroundColor: '#0d1117',
                  color: '#c9d1d9',
                  padding: '1rem',
                  borderRadius: '6px',
                  border: '1px solid #30363d',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {executionLogs.map((log, index) => (
                    // eslint-disable-next-line react/no-danger -- HTML is sanitized by ansi-to-html with escapeXML enabled
                    <span
                      key={index}
                      dangerouslySetInnerHTML={{
                        __html: convertRef.current.toHtml(log.message || '')
                      }}
                    />
                  ))}
                  {isDryRunning && (
                    <div style={{ color: '#58a6ff', marginTop: '0.5rem' }}>
                      ⏳ Execution in progress...
                    </div>
                  )}
                  <div ref={logsEndRef} />
                </div>
              </ExpandableSection>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {dryRunResult && !isDryRunning && (
        <>
          {/* Success Summary */}
          <GridItem span={12}>
            <Alert
              variant={AlertVariant.success}
              isInline
              title="Dry run completed successfully"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <CheckCircleIcon size="lg" />
                <div>
                  <strong>Execution time:</strong> {dryRunResult.execution_time_seconds?.toFixed(2)}s | 
                  <strong> Sample size:</strong> {dryRunResult.sample_size} of {dryRunResult.original_dataset_size} samples
                </div>
              </div>
            </Alert>
          </GridItem>

          {/* Execution Details */}
          <GridItem span={12}>
            <Card>
              <CardTitle>
                <Title headingLevel="h2" size="xl">
                  Execution Results
                </Title>
              </CardTitle>
              <CardBody>
                <ExpandableSection
                  toggleText="Block Execution Details"
                  isExpanded={isResultsExpanded}
                  onToggle={() => setIsResultsExpanded(!isResultsExpanded)}
                >
                  {dryRunResult.blocks_executed?.map((block, index) => (
                    <Card key={index} style={{ marginBottom: '1rem', border: '2px solid #0066cc' }}>
                      <CardBody>
                        {/* Processing Input Header */}
                        <div style={{ marginBottom: '1rem', padding: '0.5rem', background: '#e7f1fa', borderRadius: '4px' }}>
                          <div style={{ color: '#0066cc', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                            📊 Processing Input Data
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Block Type:</strong> {block.block_type}
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Input Rows:</strong> {block.input_rows?.toLocaleString() || 0}
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Input Columns:</strong> {block.output_columns?.length || 0}
                          </div>
                          {block.output_columns && (
                            <div style={{ fontSize: '0.875rem' }}>
                              <strong>Column Names:</strong> {block.output_columns.join(', ')}
                            </div>
                          )}
                        </div>

                        {/* Execution Progress */}
                        <div style={{ marginBottom: '1rem' }}>
                          <Progress
                            value={100}
                            title={block.block_name}
                            measureLocation={ProgressMeasureLocation.top}
                            style={{ height: '12px' }}
                          />
                        </div>

                        {/* Processing Complete Footer */}
                        <div style={{ padding: '0.5rem', background: '#f0f8f0', borderRadius: '4px', borderLeft: '4px solid #3e8635' }}>
                          <div style={{ color: '#3e8635', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                            ✅ Processing Complete
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Rows:</strong> {block.input_rows?.toLocaleString() || 0} → {block.output_rows?.toLocaleString() || 0}
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Columns:</strong> {block.output_columns?.length || 0}
                          </div>
                          <div style={{ fontSize: '0.875rem' }}>
                            <strong>Execution Time:</strong> {block.execution_time_seconds?.toFixed(3)}s
                          </div>
                          {block.parameters_used && Object.keys(block.parameters_used).length > 0 && (
                            <div style={{ fontSize: '0.875rem', marginTop: '0.25rem' }}>
                              <strong>Parameters:</strong> {JSON.stringify(block.parameters_used)}
                            </div>
                          )}
                        </div>
                      </CardBody>
                    </Card>
                  ))}
                </ExpandableSection>

                <div style={{ marginTop: '1.5rem' }}>
                  <Title headingLevel="h3" size="md" style={{ marginBottom: '0.5rem' }}>
                    Output Dataset Summary
                  </Title>
                  <DescriptionList isHorizontal isCompact>
                    <DescriptionListGroup>
                      <DescriptionListTerm>Total Rows</DescriptionListTerm>
                      <DescriptionListDescription>
                        {dryRunResult.final_dataset?.rows || 0}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                    <DescriptionListGroup>
                      <DescriptionListTerm>Total Columns</DescriptionListTerm>
                      <DescriptionListDescription>
                        {dryRunResult.final_dataset?.columns?.length || 0}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                    <DescriptionListGroup>
                      <DescriptionListTerm>Execution Time</DescriptionListTerm>
                      <DescriptionListDescription>
                        {dryRunResult.execution_time_seconds?.toFixed(2)}s
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  </DescriptionList>
                </div>

                {/* Time Estimation Info */}
                {enableTimeEstimation && (
                  <Alert
                    variant={AlertVariant.info}
                    isInline
                    title="Time Estimation"
                    style={{ marginTop: '1.5rem' }}
                  >
                    Time estimation details are displayed in the backend logs. 
                    Check the terminal output for detailed estimation breakdown.
                  </Alert>
                )}
              </CardBody>
            </Card>
          </GridItem>
        </>
      )}

      {/* Empty State - No Results Yet */}
      {!dryRunResult && !isDryRunning && (
        <GridItem span={12}>
          <Card>
            <CardBody>
              <EmptyState>
                <EmptyStateIcon icon={PlayIcon} />
                <Title headingLevel="h4" size="lg">
                  Ready to test your configuration
                </Title>
                <EmptyStateBody>
                  Configure the parameters above and click "Run Dry Run" to test your flow 
                  with a small sample of data. This will help you validate your settings and 
                  estimate execution time for the full dataset.
                </EmptyStateBody>
              </EmptyState>
            </CardBody>
          </Card>
        </GridItem>
      )}
    </Grid>
  );
};

export default DryRunStep;

