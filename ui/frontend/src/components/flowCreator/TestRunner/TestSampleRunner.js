import React, { useState } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  Title,
  ExpandableSection,
  CodeBlock,
  CodeBlockCode,
  Label,
  LabelGroup,
  Divider,
  Progress,
  ProgressMeasureLocation,
  Card,
  CardBody,
  CardTitle,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  InProgressIcon,
  OutlinedClockIcon,
} from '@patternfly/react-icons';

/**
 * Test Sample Runner Component
 * 
 * Displays the results of running a flow test on sample data.
 * Shows intermediate outputs at each node for debugging.
 */
const TestSampleRunner = ({
  isOpen,
  onClose,
  testConfig,
  nodes,
  edges,
  onRunTest,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [expandedResults, setExpandedResults] = useState([]);

  /**
   * Run the test through each node
   * 
   * TODO: This is a placeholder with simulated execution. The actual test
   * runner uses SSE-based step-by-step execution in VisualFlowEditor.js
   * via the /api/flow/test-step-by-step endpoint. This component is not
   * currently used in the main test flow. It should either be wired up
   * to call onRunTest or removed if no longer needed.
   */
  const runTest = async () => {
    setIsRunning(true);
    setError(null);
    setResults([]);
    setCurrentStep(0);

    try {
      const executionResults = [];
      
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        setCurrentStep(i + 1);

        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Add result for this node
        executionResults.push({
          nodeId: node.id,
          nodeName: node.label || node.config?.block_name || node.type,
          nodeType: node.type,
          status: 'success',
          output: `Sample output for ${node.type} node`,
          duration: Math.floor(Math.random() * 2000) + 500,
        });

        setResults([...executionResults]);
      }

      setExpandedResults([nodes[nodes.length - 1]?.id]); // Expand last result
    } catch (err) {
      setError(err.message || 'Test execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  /**
   * Get status icon for a result
   */
  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon color="#5ba352" />;
      case 'error':
        return <ExclamationCircleIcon color="#c9190b" />;
      case 'running':
        return <InProgressIcon color="#0066cc" />;
      default:
        return <OutlinedClockIcon color="#6a6e73" />;
    }
  };

  /**
   * Toggle result expansion
   */
  const toggleExpanded = (nodeId) => {
    setExpandedResults(prev =>
      prev.includes(nodeId)
        ? prev.filter(id => id !== nodeId)
        : [...prev, nodeId]
    );
  };

  const progress = nodes.length > 0 
    ? Math.round((currentStep / nodes.length) * 100) 
    : 0;

  return (
    <Modal
      variant={ModalVariant.large}
      title="Flow Test Results"
      isOpen={isOpen}
      onClose={onClose}
      actions={[
        <Button
          key="run"
          variant="primary"
          onClick={runTest}
          isDisabled={isRunning}
          icon={isRunning ? <Spinner size="sm" /> : null}
        >
          {isRunning ? 'Running...' : results.length > 0 ? 'Run Again' : 'Start Test'}
        </Button>,
        <Button key="close" variant="link" onClick={onClose}>
          Close
        </Button>,
      ]}
    >
      {/* Test Configuration Summary */}
      <Card isFlat style={{ marginBottom: '16px' }}>
        <CardTitle>Test Configuration</CardTitle>
        <CardBody>
          <LabelGroup>
            <Label color="blue">
              Provider: {testConfig?.modelConfig?.provider || 'Not set'}
            </Label>
            <Label color="blue">
              Model: {testConfig?.modelConfig?.model || 'Not set'}
            </Label>
            <Label color="grey">
              {nodes.length} nodes in flow
            </Label>
          </LabelGroup>
        </CardBody>
      </Card>

      {/* Progress Bar */}
      {isRunning && (
        <div style={{ marginBottom: '16px' }}>
          <Progress
            value={progress}
            title="Execution progress"
            measureLocation={ProgressMeasureLocation.top}
            label={`Step ${currentStep} of ${nodes.length}`}
          />
        </div>
      )}

      {/* Error Alert */}
      {error && (
        <Alert
          variant={AlertVariant.danger}
          isInline
          title="Test Failed"
          style={{ marginBottom: '16px' }}
        >
          {error}
        </Alert>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div>
          <Title headingLevel="h4" size="md" style={{ marginBottom: '16px' }}>
            Execution Results
          </Title>

          {results.map((result, index) => (
            <div key={result.nodeId} style={{ marginBottom: '8px' }}>
              <ExpandableSection
                toggleText={
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {getStatusIcon(result.status)}
                    <span style={{ fontWeight: 500 }}>
                      {index + 1}. {result.nodeName}
                    </span>
                    <Label isCompact color="grey">
                      {result.nodeType}
                    </Label>
                    <span style={{ fontSize: '12px', color: '#6a6e73' }}>
                      ({result.duration}ms)
                    </span>
                  </div>
                }
                isExpanded={expandedResults.includes(result.nodeId)}
                onToggle={() => toggleExpanded(result.nodeId)}
              >
                <div style={{ 
                  padding: '12px', 
                  background: '#f5f5f5', 
                  borderRadius: '4px',
                  marginTop: '8px',
                }}>
                  {result.status === 'error' ? (
                    <Alert variant={AlertVariant.danger} isInline title="Error">
                      {result.error || 'Unknown error occurred'}
                    </Alert>
                  ) : (
                    <CodeBlock>
                      <CodeBlockCode style={{ maxHeight: '200px', overflow: 'auto' }}>
                        {typeof result.output === 'object' 
                          ? JSON.stringify(result.output, null, 2)
                          : result.output
                        }
                      </CodeBlockCode>
                    </CodeBlock>
                  )}
                </div>
              </ExpandableSection>
            </div>
          ))}

          {/* Final Output Summary */}
          {!isRunning && results.length === nodes.length && (
            <>
              <Divider style={{ margin: '16px 0' }} />
              <Alert
                variant={results.every(r => r.status === 'success') 
                  ? AlertVariant.success 
                  : AlertVariant.warning
                }
                isInline
                title={results.every(r => r.status === 'success')
                  ? 'Test completed successfully!'
                  : 'Test completed with some issues'
                }
              >
                Processed {results.length} nodes in {
                  results.reduce((sum, r) => sum + r.duration, 0)
                }ms total.
              </Alert>
            </>
          )}
        </div>
      )}

      {/* Empty State */}
      {!isRunning && results.length === 0 && !error && (
        <div style={{ 
          textAlign: 'center', 
          padding: '48px',
          color: '#6a6e73',
        }}>
          <OutlinedClockIcon style={{ fontSize: '48px', marginBottom: '16px' }} />
          <Title headingLevel="h4" size="md">
            Ready to Test
          </Title>
          <p>Click "Start Test" to run your flow with the configured sample data.</p>
        </div>
      )}
    </Modal>
  );
};

export default TestSampleRunner;
