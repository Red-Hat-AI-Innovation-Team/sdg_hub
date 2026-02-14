import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Alert,
  AlertVariant,
  Grid,
  GridItem,
  Card,
  CardTitle,
  CardBody,
  Checkbox,
  List,
  ListItem,
  Title,
  Progress,
  ProgressMeasureLocation,
  Badge,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  EmptyState,
  EmptyStateIcon,
  EmptyStateHeader,
  EmptyStateBody,
  Divider,
  Sidebar,
  SidebarPanel,
  SidebarContent,
} from '@patternfly/react-core';
import { 
  CheckCircleIcon, 
  InProgressIcon, 
  PendingIcon,
  MonitoringIcon,
  SearchIcon,
} from '@patternfly/react-icons';

/**
 * Multi-Flow Monitoring Modal
 * 
 * Side-by-side layout with checkboxes on left and live monitoring cards on right
 */
const MultiFlowMonitoringModal = ({ 
  isOpen, 
  onClose, 
  selectedFlows = [], 
  executionStates = {},
  configurations = []
}) => {
  const [flowsToMonitor, setFlowsToMonitor] = useState([]);
  const [error, setError] = useState(null);
  const MAX_FLOWS = 4;

  /**
   * Reset selection when modal opens
   */
  useEffect(() => {
    if (isOpen) {
      setFlowsToMonitor([]);
      setError(null);
    }
  }, [isOpen]);

  /**
   * Toggle flow selection
   */
  const toggleFlowSelection = (flowId) => {
    setFlowsToMonitor(prev => {
      if (prev.includes(flowId)) {
        return prev.filter(id => id !== flowId);
      } else {
        if (prev.length >= MAX_FLOWS) {
          setError(`You can only monitor up to ${MAX_FLOWS} flows at once`);
          return prev;
        }
        setError(null);
        return [...prev, flowId];
      }
    });
  };

  /**
   * Get flow configuration by ID
   */
  const getFlowConfig = (flowId) => {
    return configurations.find(c => c.id === flowId);
  };

  /**
   * Parse execution state logs to extract monitoring data
   */
  const parseMonitoringData = (executionState) => {
    if (!executionState || !executionState.rawOutput) {
      return {
        overallProgress: 0,
        currentBlock: null,
        completedBlocks: 0,
        totalBlocks: 0,
        blockProgress: [],
        isComplete: false,
      };
    }

    const logs = executionState.rawOutput.split('\n');
    let blocks = [];
    let current = null;
    let blockMap = new Map();
    let flowTotalBlocks = 0;
    let isFlowComplete = false;
    
    logs.forEach(line => {
      // Detect flow start with total block count
      const flowStartMatch = line.match(/Starting flow.*?across (\d+) blocks/);
      if (flowStartMatch) {
        flowTotalBlocks = parseInt(flowStartMatch[1]);
      }
      
      // Detect block execution start
      const blockStartMatch = line.match(/Executing block (\d+)\/(\d+):\s*([\w_]+)\s*\(?([\w]+)?\)?/);
      if (blockStartMatch) {
        const blockName = blockStartMatch[3];
        const blockType = blockStartMatch[4] || 'Unknown';
        flowTotalBlocks = parseInt(blockStartMatch[2]);
        
        current = { name: blockName, type: blockType, status: 'running' };
        if (!blockMap.has(blockName)) {
          blockMap.set(blockName, current);
          blocks.push(current);
        }
      }
      
      // Detect block completion
      const blockCompleteMatch = line.match(/Block '([\w_]+)' completed/);
      if (blockCompleteMatch) {
        const blockName = blockCompleteMatch[1];
        if (blockMap.has(blockName)) {
          blockMap.get(blockName).status = 'completed';
        }
      }
      
      // Detect flow completion
      if (line.includes('completed successfully') && line.includes('final samples')) {
        isFlowComplete = true;
        current = null;
      }
      
      if (line.includes('✅ Generation completed!')) {
        isFlowComplete = true;
        current = null;
      }
    });
    
    // If flow is complete, mark everything as done
    if (isFlowComplete || (!executionState.isRunning && executionState.result)) {
      blocks.forEach(b => b.status = 'completed');
    }
    
    const completedBlocks = blocks.filter(b => b.status === 'completed').length;
    const total = flowTotalBlocks || blocks.length;
    let progress = 0;
    
    if (total > 0) {
      progress = (completedBlocks / total) * 100;
      if (isFlowComplete || (!executionState.isRunning && executionState.result)) {
        progress = 100;
      }
    }
    
    return {
      overallProgress: progress,
      currentBlock: current,
      completedBlocks: isFlowComplete ? total : completedBlocks,
      totalBlocks: total,
      blockProgress: blocks,
      isComplete: isFlowComplete || progress === 100,
    };
  };

  /**
   * Render compact flow monitoring card
   */
  const renderFlowMonitoringCard = (flowId) => {
    const config = getFlowConfig(flowId);
    const executionState = executionStates[flowId];
    const monitoringData = parseMonitoringData(executionState);
    
    if (!config) return null;

    return (
      <Card isCompact isFullHeight>
        <CardTitle>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <Title headingLevel="h3" size="md">
                {config.flow_name || config.name}
              </Title>
              <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginTop: '4px' }}>
                {config.model_configuration?.model || 'No model'}
              </div>
            </div>
            {executionState?.isRunning && (
              <Badge style={{ background: '#0066cc', color: 'white' }}>
                <InProgressIcon /> Running
              </Badge>
            )}
            {monitoringData.isComplete && (
              <Badge style={{ background: '#3e8635', color: 'white' }}>
                <CheckCircleIcon /> Complete
              </Badge>
            )}
          </div>
        </CardTitle>
        <CardBody>
          {!executionState || (!executionState.isRunning && !executionState.rawOutput) ? (
            <EmptyState variant="xs">
              <EmptyStateHeader
                titleText="Not running"
                icon={<EmptyStateIcon icon={PendingIcon} />}
                headingLevel="h4"
              />
              <EmptyStateBody>
                This flow is not currently executing
              </EmptyStateBody>
            </EmptyState>
          ) : (
            <div>
              {/* Overall Progress */}
              <Progress
                value={monitoringData.overallProgress}
                title="Progress"
                size="sm"
                measureLocation={ProgressMeasureLocation.top}
                style={{ marginBottom: '1rem' }}
              />
              
              {/* Stats */}
              <DescriptionList isHorizontal isCompact style={{ fontSize: '0.875rem' }}>
                <DescriptionListGroup>
                  <DescriptionListTerm>Blocks</DescriptionListTerm>
                  <DescriptionListDescription>
                    {monitoringData.completedBlocks} / {monitoringData.totalBlocks}
                  </DescriptionListDescription>
                </DescriptionListGroup>
                <DescriptionListGroup>
                  <DescriptionListTerm>Current</DescriptionListTerm>
                  <DescriptionListDescription>
                    {monitoringData.isComplete ? (
                      <span style={{ color: '#3e8635' }}>✅ Complete</span>
                    ) : monitoringData.currentBlock ? (
                      monitoringData.currentBlock.name
                    ) : (
                      'Initializing...'
                    )}
                  </DescriptionListDescription>
                </DescriptionListGroup>
              </DescriptionList>
              
              <Divider style={{ margin: '0.75rem 0' }} />
              
              {/* Block Progress List - Compact */}
              {monitoringData.blockProgress.length > 0 && (
                <div>
                  <div style={{ 
                    fontSize: '0.75rem', 
                    fontWeight: 'bold', 
                    marginBottom: '0.5rem',
                    color: '#6a6e73'
                  }}>
                    Block Status
                  </div>
                  <List 
                    isPlain 
                    isBordered 
                    style={{ 
                      maxHeight: '200px', 
                      overflowY: 'auto',
                      fontSize: '0.75rem'
                    }}
                  >
                    {monitoringData.blockProgress.map((block, index) => (
                      <ListItem key={index}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '0.25rem 0.5rem',
                          background: block.status === 'running' ? '#e7f1fa' : 
                                      block.status === 'completed' ? '#f0f8f0' : '#f5f5f5'
                        }}>
                          <div>
                            <strong>{block.name}</strong>
                          </div>
                          <div>
                            {block.status === 'completed' && <CheckCircleIcon color="#3e8635" style={{ fontSize: '0.875rem' }} />}
                            {block.status === 'running' && <InProgressIcon color="#0066cc" style={{ fontSize: '0.875rem' }} />}
                            {block.status === 'pending' && <PendingIcon color="#6a6e73" style={{ fontSize: '0.875rem' }} />}
                          </div>
                        </div>
                      </ListItem>
                    ))}
                  </List>
                </div>
              )}
            </div>
          )}
        </CardBody>
      </Card>
    );
  };

  /**
   * Render flow selection sidebar
   */
  const renderFlowSelectionSidebar = () => {
    if (selectedFlows.length === 0) {
      return (
        <div style={{ padding: '1rem' }}>
          <Alert
            variant={AlertVariant.warning}
            title="No flows selected"
            isInline
            isPlain
          >
            Select flows from the list first
          </Alert>
        </div>
      );
    }

    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRight: '1px solid var(--pf-v5-global--BorderColor--100)'
      }}>
        <div style={{ 
          padding: '1rem', 
          borderBottom: '1px solid var(--pf-v5-global--BorderColor--100)',
          background: 'var(--pf-v5-global--BackgroundColor--200)'
        }}>
          <Title headingLevel="h3" size="md">
            Select Flows to Monitor
          </Title>
          <div style={{ 
            fontSize: '0.875rem', 
            color: 'var(--pf-v5-global--Color--200)',
            marginTop: '0.25rem'
          }}>
            {flowsToMonitor.length} of {MAX_FLOWS} selected
          </div>
        </div>
        
        <div style={{ 
          flex: 1, 
          overflowY: 'auto', 
          padding: '0.5rem' 
        }}>
          <List isPlain>
            {selectedFlows.map(flowId => {
              const config = getFlowConfig(flowId);
              if (!config) return null;
              
              const isSelected = flowsToMonitor.includes(flowId);
              const isRunning = executionStates[flowId]?.isRunning;
              
              return (
                <ListItem key={flowId} style={{ padding: '0.5rem' }}>
                  <Checkbox
                    id={`monitor-checkbox-${flowId}`}
                    label={
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '6px',
                          flexWrap: 'wrap'
                        }}>
                          <span style={{ 
                            fontWeight: isSelected ? 'bold' : 'normal',
                            fontSize: '0.875rem'
                          }}>
                            {config.flow_name || config.name}
                          </span>
                          {isRunning && (
                            <Badge style={{ background: '#0066cc', color: 'white', fontSize: '0.75rem' }}>
                              ● Live
                            </Badge>
                          )}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                          {config.model_configuration?.model || 'No model'}
                        </div>
                      </div>
                    }
                    isChecked={isSelected}
                    onChange={() => toggleFlowSelection(flowId)}
                  />
                </ListItem>
              );
            })}
          </List>
        </div>
      </div>
    );
  };

  /**
   * Render monitoring content area
   */
  const renderMonitoringContent = () => {
    if (flowsToMonitor.length === 0) {
      return (
        <EmptyState variant="lg">
          <EmptyStateHeader
            titleText="Select flows to monitor"
            icon={<EmptyStateIcon icon={SearchIcon} />}
            headingLevel="h2"
          />
          <EmptyStateBody>
            Use the checkboxes on the left to select up to {MAX_FLOWS} flows.
            Monitoring cards will appear here in real-time as you select them.
          </EmptyStateBody>
        </EmptyState>
      );
    }

    return (
      <div style={{ padding: '1rem' }}>
        {error && (
          <Alert
            variant={AlertVariant.warning}
            title={error}
            isInline
            style={{ marginBottom: '1rem' }}
          />
        )}

        <Alert
          variant={AlertVariant.info}
          title={`Monitoring ${flowsToMonitor.length} flow${flowsToMonitor.length > 1 ? 's' : ''} in real-time`}
          isInline
          isPlain
          style={{ marginBottom: '1rem' }}
        >
          Cards update automatically as flows execute. Click a flow name in the main list for detailed terminal output.
        </Alert>

        <Grid hasGutter>
          {flowsToMonitor.map(flowId => (
            <GridItem
              key={flowId}
              span={12}
              sm={flowsToMonitor.length === 1 ? 12 : 6}
              md={flowsToMonitor.length === 1 ? 12 : flowsToMonitor.length === 2 ? 6 : 6}
              lg={flowsToMonitor.length === 1 ? 12 : flowsToMonitor.length === 2 ? 6 : 6}
            >
              {renderFlowMonitoringCard(flowId)}
            </GridItem>
          ))}
        </Grid>
      </div>
    );
  };

  return (
    <Modal
      variant={ModalVariant.large}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MonitoringIcon />
          <span>Multi-Flow Monitoring</span>
        </div>
      }
      isOpen={isOpen}
      onClose={onClose}
      actions={[
        <Button key="close" variant="primary" onClick={onClose}>
          Close
        </Button>
      ]}
    >
      <div style={{ 
        display: 'flex',
        height: '70vh',
        minHeight: '500px',
        margin: '-1.5rem'
      }}>
        {/* Left Sidebar - Flow Selection */}
        <div style={{ 
          width: '300px', 
          flexShrink: 0,
          background: 'var(--pf-v5-global--BackgroundColor--100)'
        }}>
          {renderFlowSelectionSidebar()}
        </div>

        {/* Right Content - Monitoring Cards */}
        <div style={{ 
          flex: 1, 
          overflowY: 'auto',
          background: 'var(--pf-v5-global--BackgroundColor--100)'
        }}>
          {renderMonitoringContent()}
        </div>
      </div>
    </Modal>
  );
};

export default MultiFlowMonitoringModal;

