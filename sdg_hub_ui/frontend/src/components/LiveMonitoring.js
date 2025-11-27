import React, { useState, useEffect, useRef } from 'react';
import {
  Grid,
  GridItem,
  Card,
  CardTitle,
  CardBody,
  Title,
  Progress,
  ProgressMeasureLocation,
  Alert,
  AlertVariant,
  List,
  ListItem,
  Badge,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  ExpandableSection,
  CodeBlock,
  CodeBlockCode,
} from '@patternfly/react-core';
import { CheckCircleIcon, InProgressIcon, PendingIcon } from '@patternfly/react-icons';

/**
 * Strip ANSI escape codes from a string
 */
const stripAnsi = (str) => {
  if (!str) return '';
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[([0-9;]*)m/g, '');
};

/**
 * Live Monitoring Page
 * 
 * Displays real-time progress during flow.generate() execution:
 * - Overall flow progress
 * - Block-by-block status
 * - LLM request tracking
 * - Time metrics
 */
const LiveMonitoring = ({ generationLogs, isGenerating }) => {
  const [blockProgress, setBlockProgress] = useState([]);
  const [currentBlock, setCurrentBlock] = useState(null);
  const [requestStats, setRequestStats] = useState({});
  const [overallProgress, setOverallProgress] = useState(0);
  const [lastProcessedIndex, setLastProcessedIndex] = useState(0);
  const [totalBlocks, setTotalBlocks] = useState(0);
  const [completedBlocks, setCompletedBlocks] = useState(0);
  const [tokenStats, setTokenStats] = useState({
    total: { prompt: 0, completion: 0, total: 0 },
    byBlock: {}
  });
  
  // Use ref to preserve request totals across re-renders
  const requestTotalsRef = useRef({});

  /**
   * Reset state on mount (when key changes, component remounts)
   */
  useEffect(() => {
    setBlockProgress([]);
    setCurrentBlock(null);
    setRequestStats({});
    setOverallProgress(0);
    setLastProcessedIndex(0);
    setTotalBlocks(0);
    setCompletedBlocks(0);
    setTokenStats({ total: { prompt: 0, completion: 0, total: 0 }, byBlock: {} });
    requestTotalsRef.current = {};
  }, []); // Empty deps - runs only on mount

  /**
   * Parse logs to extract progress information
   * Re-runs whenever new logs arrive
   */
  useEffect(() => {
    if (!generationLogs || generationLogs.length === 0) {
      return;
    }

    // Check if generation is complete (simple check)
    const hasCompletionMessage = generationLogs.some(log => 
      (log.type === 'complete') ||
      (log.type === 'success' && log.message && log.message.includes('Generation completed')) ||
      (log.message && log.message.includes('completed successfully') && log.message.includes('final samples')) ||
      (log.message && log.message.includes('Generation completed!')) ||
      (log.message && log.message.match(/Generation completed!.*samples.*columns/))
    );
    

    // Parse ALL logs to build complete state from scratch
    let blocks = [];
    let current = null;
    let requests = {};
    let blockMap = new Map(); // Track unique blocks
    let tokens = { total: { prompt: 0, completion: 0, total: 0 }, byBlock: {} };
    
    let flowTotalBlocks = 0;
    let flowCompletedBlocks = 0;
    let isFlowComplete = false;
    
    generationLogs.forEach(log => {
      const msg = stripAnsi(log.message || '');
      
      // Detect completion from log type (backend sends type: 'complete')
      if (log.type === 'complete' || (log.type === 'success' && msg.includes('Generation completed'))) {
        isFlowComplete = true;
        current = null;
      }
      
      // Detect flow start with total block count
      // Format: "Starting flow 'Name' v1.0.0 with X samples across Y blocks"
      const flowStartMatch = msg.match(/Starting flow.*?across (\d+) blocks/);
      if (flowStartMatch) {
        flowTotalBlocks = parseInt(flowStartMatch[1]);
      }
      
      // Detect block execution start (handles INFO prefix and line breaks)
      const blockStartMatch = msg.match(/Executing block (\d+)\/(\d+):\s*([\w_]+)\s*\(?([\w]+)?\)?/);
      if (blockStartMatch) {
        const blockName = blockStartMatch[3];
        const blockType = blockStartMatch[4] || 'Unknown';
        const blockNum = parseInt(blockStartMatch[1]);
        flowTotalBlocks = parseInt(blockStartMatch[2]); // Update total from current block execution
        
        current = { name: blockName, type: blockType, status: 'running' };
        if (!blockMap.has(blockName)) {
          blockMap.set(blockName, current);
          blocks.push(current);
        }
      }
      
      // Detect block completion
      const blockCompleteMatch = msg.match(/Block '([\w_]+)' completed/);
      if (blockCompleteMatch) {
        const blockName = blockCompleteMatch[1];
        if (blockMap.has(blockName)) {
          blockMap.get(blockName).status = 'completed';
        }
      }
      
      // Detect flow completion from backend logs - multiple patterns
      if (msg.includes('completed successfully') && msg.includes('final samples')) {
        isFlowComplete = true;
        current = null;
        blockMap.forEach(block => {
          block.status = 'completed';
        });
        flowCompletedBlocks = flowTotalBlocks;
      }
      
      // Also detect completion from the type: 'complete' message
      if (log.type === 'complete') {
        isFlowComplete = true;
        current = null;
        blockMap.forEach(block => {
          block.status = 'completed';
        });
        flowCompletedBlocks = flowTotalBlocks;
      }
      
      // Detect completion from "Generation completed!" message
      if (msg.includes('Generation completed!')) {
        isFlowComplete = true;
        current = null;
        blockMap.forEach(block => {
          block.status = 'completed';
        });
        flowCompletedBlocks = flowTotalBlocks;
      }
      
      // Detect tqdm progress bars
      // Format: [block_name] LLM Requests:  56%|#####     | 9/16 [00:08<00:06, 1.12req/s]
      const tqdmMatch = msg.match(/\[([\w_]+)\]\s+LLM Requests:\s+(\d+)%\|.*?\|\s*(\d+)\/(\d+)/);
      if (tqdmMatch) {
        const [_, blockName, percent, completed, total] = tqdmMatch;
        
        // Store total in ref on first detection (it shouldn't change)
        if (!requestTotalsRef.current[blockName]) {
          requestTotalsRef.current[blockName] = parseInt(total);
        }
        
        // Always use the stored total
        requests[blockName] = {
          completed: parseInt(completed),
          total: requestTotalsRef.current[blockName],
          percent: parseInt(percent)
        };
      }
      
      // Parse TOKEN_USAGE messages
      // Format: 🔢 [block_name] Tokens → in: 1,234 | out: 567 | total: 1,801
      const tokenMatch = msg.match(/🔢 \[([\w_]+)\] Tokens → in: ([\d,]+) \| out: ([\d,]+) \| total: ([\d,]+)/);
      if (tokenMatch) {
        const [_, blockName, prompt, completion, total] = tokenMatch;
        
        if (!tokens.byBlock[blockName]) {
          tokens.byBlock[blockName] = { prompt: 0, completion: 0, total: 0 };
        }
        
        // Remove commas and parse numbers
        tokens.byBlock[blockName].prompt += parseInt(prompt.replace(/,/g, ''));
        tokens.byBlock[blockName].completion += parseInt(completion.replace(/,/g, ''));
        tokens.byBlock[blockName].total += parseInt(total.replace(/,/g, ''));
      }
    });
    
    // Calculate total tokens
    Object.values(tokens.byBlock).forEach(block => {
      tokens.total.prompt += block.prompt;
      tokens.total.completion += block.completion;
      tokens.total.total += block.total;
    });
    
    // If flow is complete, mark everything as done
    if (isFlowComplete || (!isGenerating && hasCompletionMessage)) {
      current = null;
      blocks.forEach(b => b.status = 'completed');
    }
    
    // Count completed blocks
    let actualCompletedBlocks = blocks.filter(b => b.status === 'completed').length;
    
    // If flow complete, set to total
    if (isFlowComplete || (!isGenerating && hasCompletionMessage)) {
      actualCompletedBlocks = flowTotalBlocks || blocks.length;
    }
    
    // Update state
    setBlockProgress(blocks);
    setCurrentBlock(current);
    setRequestStats(requests);
    setTotalBlocks(flowTotalBlocks || blocks.length);
    setCompletedBlocks(actualCompletedBlocks);
    setTokenStats(tokens);
    
    // Calculate overall progress using actual counts
    const total = flowTotalBlocks || blocks.length;
    let finalProgress = 0;
    if (total > 0) {
      finalProgress = (actualCompletedBlocks / total) * 100;
      
      // Force 100% if we detected completion
      if (isFlowComplete || (!isGenerating && hasCompletionMessage)) {
        finalProgress = 100;
        actualCompletedBlocks = total; // Also force completed blocks to match total
      }
    }
    
    // Update completed blocks again after forcing completion
    setCompletedBlocks(actualCompletedBlocks);
    setOverallProgress(finalProgress);
  }, [generationLogs, isGenerating]); // Depend on both logs and generating state

  if (!isGenerating && (!generationLogs || generationLogs.length === 0)) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <Alert
          variant={AlertVariant.info}
          isInline
          title="No active generation"
        >
          <p>
            Start a generation from the <strong>Generate Data</strong> page to see live monitoring here.
          </p>
        </Alert>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '2rem', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
      <Grid hasGutter>
        {/* Overall Progress */}
        <GridItem span={12}>
          <Card>
            <CardTitle>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title headingLevel="h2" size="xl">
                  Overall Progress
                </Title>
                {isGenerating && (
                  <div style={{ 
                    padding: '0.5rem 1rem', 
                    background: '#0066cc', 
                    borderRadius: '4px',
                  }}>
                    <span style={{ color: '#ffffff', fontWeight: 'bold' }}>
                      ● LIVE
                    </span>
                    <span style={{ marginLeft: '0.5rem', fontSize: '0.875rem', color: '#ffffff' }}>
                      ({generationLogs.length} log entries)
                    </span>
                  </div>
                )}
              </div>
            </CardTitle>
            <CardBody>
              <Progress
                value={overallProgress}
                title="Flow Execution"
                size="lg"
                measureLocation={ProgressMeasureLocation.top}
              />
              <div style={{ marginTop: '1rem' }}>
                <DescriptionList isHorizontal isCompact>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Blocks Completed</DescriptionListTerm>
                    <DescriptionListDescription>
                      {completedBlocks} / {totalBlocks || blockProgress.length}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Current Block</DescriptionListTerm>
                    <DescriptionListDescription>
                      {overallProgress === 100 ? (
                        <span style={{ color: '#3e8635', fontWeight: 'bold' }}>✅ All Complete</span>
                      ) : currentBlock ? (
                        currentBlock.name
                      ) : (
                        'Initializing...'
                      )}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Status</DescriptionListTerm>
                    <DescriptionListDescription>
                      {isGenerating ? (
                        <Badge><InProgressIcon /> Running</Badge>
                      ) : overallProgress >= 99 ? (
                        <Badge style={{ background: '#3e8635', color: 'white' }}><CheckCircleIcon /> Completed</Badge>
                      ) : (
                        <Badge><CheckCircleIcon /> Stopped</Badge>
                      )}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  {tokenStats.total.total > 0 && (
                    <>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Total Tokens</DescriptionListTerm>
                        <DescriptionListDescription>
                          <strong style={{ color: '#0066cc' }}>{tokenStats.total.total.toLocaleString()}</strong>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Input Tokens</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span style={{ color: '#3e8635' }}>{tokenStats.total.prompt.toLocaleString()}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Output Tokens</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span style={{ color: '#a30000' }}>{tokenStats.total.completion.toLocaleString()}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                    </>
                  )}
                </DescriptionList>
              </div>
            </CardBody>
          </Card>
        </GridItem>

        {/* Block Status Timeline */}
        <GridItem span={12}>
          <Card isFullHeight>
            <CardTitle>
              <Title headingLevel="h2" size="xl">
                Block Execution Status
              </Title>
            </CardTitle>
            <CardBody>
              {overallProgress >= 99 && !isGenerating && (
                <Alert
                  variant={AlertVariant.success}
                  isInline
                  title="Flow Completed!"
                  style={{ marginBottom: '1rem' }}
                >
                  <p>✅ All {totalBlocks} blocks executed successfully!</p>
                </Alert>
              )}
              
              <List isPlain isBordered style={{ maxHeight: '500px', overflowY: 'auto' }}>
                {blockProgress.map((block, index) => (
                  <ListItem key={index}>
                    <div style={{
                      padding: '1rem',
                      background: block.status === 'running' ? '#e7f1fa' : 
                                  block.status === 'completed' ? '#f0f8f0' : '#f5f5f5'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <strong>{block.name}</strong>
                          <div style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
                            {block.type}
                          </div>
                        </div>
                        <div>
                          {block.status === 'completed' && <CheckCircleIcon color="#3e8635" />}
                          {block.status === 'running' && <InProgressIcon color="#0066cc" />}
                          {block.status === 'pending' && <PendingIcon color="#6a6e73" />}
                        </div>
                      </div>
                      
                      {/* Show request progress if available */}
                      {requestStats[block.name] && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <Progress
                            value={requestStats[block.name].percent}
                            title={`${requestStats[block.name].completed}/${requestStats[block.name].total} requests`}
                            measureLocation={ProgressMeasureLocation.top}
                            size="sm"
                          />
                        </div>
                      )}
                      
                      {/* Show token usage if available */}
                      {tokenStats.byBlock[block.name] && (
                        <div style={{ 
                          marginTop: '0.5rem', 
                          fontSize: '0.875rem', 
                          color: '#6a6e73',
                          display: 'flex',
                          gap: '1rem'
                        }}>
                          <span>
                            🔢 <strong style={{ color: '#0066cc' }}>{tokenStats.byBlock[block.name].total.toLocaleString()}</strong> tokens
                          </span>
                          <span>
                            (↑ <span style={{ color: '#3e8635' }}>{tokenStats.byBlock[block.name].prompt.toLocaleString()}</span>
                            {' '}↓ <span style={{ color: '#a30000' }}>{tokenStats.byBlock[block.name].completion.toLocaleString()}</span>)
                          </span>
                        </div>
                      )}
                    </div>
                  </ListItem>
                ))}
              </List>
            </CardBody>
          </Card>
        </GridItem>


        </Grid>
    </div>
  );
};

export default LiveMonitoring;

