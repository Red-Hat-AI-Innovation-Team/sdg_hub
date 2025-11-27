import React, { useState, useEffect } from 'react';
import {
  PageSection,
  Tabs,
  Tab,
  TabTitleText,
  Card,
  CardTitle,
  CardBody,
  Button,
  Title,
  Flex,
  FlexItem,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  Label,
  LabelGroup,
  CodeBlock,
  CodeBlockCode,
  Progress,
  ProgressMeasureLocation,
  Grid,
  GridItem,
  Badge,
  List,
  ListItem,
  Dropdown,
  DropdownList,
  DropdownItem,
  MenuToggle,
  Tooltip,
} from '@patternfly/react-core';
import { ArrowLeftIcon, InfoCircleIcon, TerminalIcon, ChartLineIcon, CheckCircleIcon, InProgressIcon, StopIcon, PlayIcon, RedoIcon, HistoryIcon, CaretDownIcon } from '@patternfly/react-icons';
import AnsiToHtml from 'ansi-to-html';
import LiveMonitoring from '../LiveMonitoring';

/**
 * Detail view for a configuration with tabs
 */
const ConfigurationDetailView = ({ 
  configuration, 
  onClose, 
  onRefresh, 
  executionState, 
  onDryRun, 
  onGenerate, 
  onGenerateFromCheckpoint,
  onClearTerminal, 
  onStop,
  checkpointInfo 
}) => {
  const [activeTabKey, setActiveTabKey] = useState('overview');
  const [isRunMenuOpen, setIsRunMenuOpen] = useState(false);
  
  // Check if flow is running
  const isRunning = executionState?.isRunning;
  
  // Check if flow is in a resumable state (failed or stopped)
  const isResumable = executionState?.status === 'failed' || 
                      executionState?.status === 'error' || 
                      executionState?.status === 'cancelled' || 
                      executionState?.status === 'stopped';
  
  // Check if checkpoints exist
  const hasCheckpoints = checkpointInfo?.has_checkpoints;

  /**
   * Render Overview tab
   */
  const renderOverview = () => {
    // Extract model name (support both old and new field names)
    const modelConfig = configuration.model_configuration || configuration.model_config || {};
    const modelName = modelConfig.model || 'Not configured';
    const apiBase = modelConfig.api_base || 'Default';
    
    // Extract dataset info (support both old and new field names)
    const datasetConfig = configuration.dataset_configuration || configuration.dataset_config || {};
    const datasetPath = datasetConfig.data_files || 'Not specified';
    const numSamples = datasetConfig.num_samples || 'All';
    const shuffle = datasetConfig.shuffle ? 'Yes' : 'No';
    
    return (
      <Card>
        <CardBody>
          <Title headingLevel="h2" size="xl" style={{ marginBottom: '24px' }}>
            Flow Configuration Details
          </Title>
          
          <DescriptionList isHorizontal columnModifier={{ default: '2Col' }}>
            <DescriptionListGroup>
              <DescriptionListTerm>Flow Name</DescriptionListTerm>
              <DescriptionListDescription>{configuration.flow_name}</DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Flow ID</DescriptionListTerm>
              <DescriptionListDescription>
                <code>{configuration.flow_id}</code>
              </DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Model</DescriptionListTerm>
              <DescriptionListDescription>{modelName}</DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>API Base</DescriptionListTerm>
              <DescriptionListDescription>{apiBase}</DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Dataset Path</DescriptionListTerm>
              <DescriptionListDescription>
                <code>{datasetPath}</code>
              </DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Number of Samples</DescriptionListTerm>
              <DescriptionListDescription>{numSamples}</DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Shuffle</DescriptionListTerm>
              <DescriptionListDescription>{shuffle}</DescriptionListDescription>
            </DescriptionListGroup>
            
            <DescriptionListGroup>
              <DescriptionListTerm>Created At</DescriptionListTerm>
              <DescriptionListDescription>
                {new Date(configuration.created_at).toLocaleString()}
              </DescriptionListDescription>
            </DescriptionListGroup>
            
            {configuration.tags && configuration.tags.length > 0 && (
              <DescriptionListGroup>
                <DescriptionListTerm>Tags</DescriptionListTerm>
                <DescriptionListDescription>
                  <LabelGroup>
                    {configuration.tags.map((tag, idx) => (
                      <Label key={idx} color="blue" isCompact>
                        {tag}
                      </Label>
                    ))}
                  </LabelGroup>
                </DescriptionListDescription>
              </DescriptionListGroup>
            )}
          </DescriptionList>
          
          <Title headingLevel="h3" size="lg" style={{ marginTop: '32px', marginBottom: '16px' }}>
            Model Configuration
          </Title>
          <CodeBlock>
            <CodeBlockCode>
              {JSON.stringify(modelConfig, null, 2)}
            </CodeBlockCode>
          </CodeBlock>
          
          <Title headingLevel="h3" size="lg" style={{ marginTop: '32px', marginBottom: '16px' }}>
            Dataset Configuration
          </Title>
          <CodeBlock>
            <CodeBlockCode>
              {JSON.stringify(datasetConfig, null, 2)}
            </CodeBlockCode>
          </CodeBlock>
        </CardBody>
      </Card>
    );
  };

  /**
   * Render Terminal tab
   */
  const renderTerminal = () => {
    const hasOutput = executionState && executionState.rawOutput;
    
    // Convert ANSI codes to HTML (same settings as original)
    const convert = new AnsiToHtml({
      fg: '#d4d4d4',
      bg: '#1e1e1e',
      newline: false,
      escapeXML: true,
    });
    
    return (
      <Card>
        <CardBody>
          <Flex direction={{ default: 'column' }} spaceItems={{ default: 'spaceItemsSm' }}>
            <FlexItem>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title headingLevel="h2" size="xl">
                  Execution Output
                </Title>
                {hasOutput && onClearTerminal && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => onClearTerminal(configuration.id)}
                    isDisabled={executionState?.isRunning}
                  >
                    Clear Terminal
                  </Button>
                )}
              </div>
            </FlexItem>
            
            {!hasOutput ? (
              <FlexItem>
                <p>Terminal output will appear here when you run Dry Run or Generate.</p>
                <p style={{ color: 'var(--pf-v5-global--Color--200)', marginTop: '16px' }}>
                  Click "Dry Run" or "Generate" button to see execution logs.
                </p>
              </FlexItem>
            ) : (
              <FlexItem>
                <div
                  id="terminal-output"
                  style={{
                    fontSize: '14px',
                    fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                    minHeight: '400px',
                    maxHeight: '800px',
                    height: 'auto',
                    overflowY: 'auto',
                    overflowX: 'auto',
                    backgroundColor: '#0d1117',
                    color: '#c9d1d9',
                    padding: '1rem',
                    borderRadius: '6px',
                    border: '1px solid #30363d',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    lineHeight: '1.5'
                  }}
                  dangerouslySetInnerHTML={{
                    __html: convert.toHtml(executionState.rawOutput)
                  }}
                />
                {executionState.isRunning && (
                  <div style={{ 
                    marginTop: '8px', 
                    color: '#58a6ff',
                    fontWeight: 'bold'
                  }}>
                    ⏳ Execution in progress...
                  </div>
                )}
              </FlexItem>
            )}
          </Flex>
        </CardBody>
      </Card>
    );
  };

  /**
   * Strip ANSI escape codes from a string
   */
  const stripAnsi = (str) => {
    if (!str) return '';
    // eslint-disable-next-line no-control-regex
    return str.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[([0-9;]*)m/g, '');
  };

  /**
   * Parse logs to extract block progress
   */
  const parseBlockProgress = () => {
    if (!executionState || !executionState.rawOutput) return [];
    
    const logs = stripAnsi(executionState.rawOutput).split('\n');
    const blocks = [];
    let currentBlock = null;
    
    logs.forEach(line => {
      // Detect block execution start
      const blockMatch = line.match(/Executing block (\d+)\/(\d+):\s*([\w_]+)/);
      if (blockMatch) {
        currentBlock = {
          name: blockMatch[3],
          number: parseInt(blockMatch[1]),
          total: parseInt(blockMatch[2]),
          status: 'running'
        };
        blocks.push(currentBlock);
      }
      
      // Detect block completion
      if (line.includes('Processing Complete') && currentBlock) {
        currentBlock.status = 'complete';
      }
    });
    
    return blocks;
  };

  /**
   * Render Live Monitoring tab
   */
  const renderLiveMonitoring = () => {
    // Convert raw output to generation logs format for LiveMonitoring component
    const generationLogs = executionState && executionState.rawOutput 
      ? executionState.rawOutput.split('\n').map((line, idx) => ({
          type: 'log',
          message: line,
          timestamp: Date.now() + idx
        }))
      : [];
    
    const isGenerating = executionState?.isRunning || false;
    
    return (
      <LiveMonitoring 
        key={executionState?.runId || 'default'} // Force remount when run changes
        generationLogs={generationLogs} 
        isGenerating={isGenerating}
      />
    );
  };

  /**
   * Automatically switch to terminal tab when execution starts or when opening a running config
   */
  useEffect(() => {
    if (executionState && executionState.isRunning) {
      setActiveTabKey('terminal');
    }
  }, [executionState?.isRunning]);

  /**
   * Auto-scroll terminal to bottom when new output arrives
   */
  useEffect(() => {
    const terminalDiv = document.getElementById('terminal-output');
    if (terminalDiv && executionState?.rawOutput) {
      terminalDiv.scrollTop = terminalDiv.scrollHeight;
    }
  }, [executionState?.rawOutput]);

  return (
    <PageSection>
      <Flex direction={{ default: 'column' }} spaceItems={{ default: 'spaceItemsLg' }}>
        <FlexItem>
          <Flex spaceItems={{ default: 'spaceItemsMd' }} alignItems={{ default: 'alignItemsCenter' }}>
            <FlexItem>
              <Button variant="link" icon={<ArrowLeftIcon />} onClick={onClose}>
                Back to Configurations
              </Button>
            </FlexItem>
            <FlexItem>
              {isRunning ? (
                // Show Stop button when running
                onStop && (
                  <Button 
                    variant="danger" 
                    icon={<StopIcon />} 
                    onClick={() => onStop(configuration)}
                  >
                    Stop
                  </Button>
                )
              ) : isResumable && hasCheckpoints ? (
                // Show dropdown with resume options for failed/stopped flows with checkpoints
                <Dropdown
                  isOpen={isRunMenuOpen}
                  onSelect={() => setIsRunMenuOpen(false)}
                  onOpenChange={(isOpen) => setIsRunMenuOpen(isOpen)}
                  toggle={(toggleRef) => (
                    <MenuToggle
                      ref={toggleRef}
                      onClick={() => setIsRunMenuOpen(!isRunMenuOpen)}
                      isExpanded={isRunMenuOpen}
                      variant="primary"
                      splitButtonOptions={{
                        variant: 'action',
                        items: [
                          <Button
                            key="run"
                            variant="primary"
                            icon={<HistoryIcon />}
                            onClick={() => {
                              setIsRunMenuOpen(false);
                              onGenerateFromCheckpoint && onGenerateFromCheckpoint(configuration);
                            }}
                          >
                            Resume from Checkpoint
                          </Button>
                        ]
                      }}
                    >
                      <CaretDownIcon />
                    </MenuToggle>
                  )}
                >
                  <DropdownList>
                    <DropdownItem
                      key="resume"
                      icon={<HistoryIcon />}
                      onClick={() => onGenerateFromCheckpoint && onGenerateFromCheckpoint(configuration)}
                    >
                      Resume from Checkpoint ({checkpointInfo?.samples_completed || 0} samples completed)
                    </DropdownItem>
                    <DropdownItem
                      key="fresh"
                      icon={<RedoIcon />}
                      onClick={() => onGenerate && onGenerate(configuration)}
                    >
                      Run from Scratch
                    </DropdownItem>
                  </DropdownList>
                </Dropdown>
              ) : (
                // Show simple Run button for configured flows without checkpoints
                onGenerate && configuration.status !== 'not_configured' && configuration.status !== 'draft' && (
                  <Button 
                    variant="primary" 
                    icon={<PlayIcon />} 
                    onClick={() => onGenerate(configuration)}
                  >
                    Run
                  </Button>
                )
              )}
            </FlexItem>
          </Flex>
        </FlexItem>
        
        <FlexItem>
          <Tabs
            activeKey={activeTabKey}
            onSelect={(event, tabKey) => setActiveTabKey(tabKey)}
            aria-label="Configuration details tabs"
            role="region"
          >
            <Tab
              eventKey="overview"
              title={<TabTitleText><InfoCircleIcon /> Overview</TabTitleText>}
              aria-label="Overview tab"
            >
              {renderOverview()}
            </Tab>
            
            <Tab
              eventKey="terminal"
              title={<TabTitleText><TerminalIcon /> Terminal</TabTitleText>}
              aria-label="Terminal tab"
            >
              {renderTerminal()}
            </Tab>
            
            <Tab
              eventKey="monitoring"
              title={<TabTitleText><ChartLineIcon /> Live Monitoring</TabTitleText>}
              aria-label="Live monitoring tab"
            >
              {renderLiveMonitoring()}
            </Tab>
          </Tabs>
        </FlexItem>
      </Flex>
    </PageSection>
  );
};

export default ConfigurationDetailView;

