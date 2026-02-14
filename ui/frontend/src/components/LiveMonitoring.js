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
  const [wasCancelled, setWasCancelled] = useState(false);
  const [tokenStats, setTokenStats] = useState({
    total: { prompt: 0, completion: 0, total: 0 },
    byBlock: {}
  });
  
  // Use ref to preserve request totals across re-renders
  const requestTotalsRef = useRef({});
  // Use ref to preserve total blocks once detected (shouldn't change during execution)
  const totalBlocksRef = useRef(0);

  // Track previous generating state to detect when generation starts
  const prevIsGeneratingRef = useRef(false);
  
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
    totalBlocksRef.current = 0;
  }, []); // Empty deps - runs only on mount
  
  /**
   * Reset state when a new generation starts
   */
  useEffect(() => {
    // Detect when generation starts (transition from not generating to generating)
    if (isGenerating && !prevIsGeneratingRef.current) {
      // New generation started - reset all state
      setBlockProgress([]);
      setCurrentBlock(null);
      setRequestStats({});
      setOverallProgress(0);
      setLastProcessedIndex(0);
      setTotalBlocks(0);
      setCompletedBlocks(0);
      setTokenStats({ total: { prompt: 0, completion: 0, total: 0 }, byBlock: {} });
      setWasCancelled(false);
      requestTotalsRef.current = {};
      totalBlocksRef.current = 0;
    }
    prevIsGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  /**
   * Parse logs to extract progress information
   * Re-runs whenever new logs arrive
   */
  useEffect(() => {
    if (!generationLogs || generationLogs.length === 0) {
      return;
    }

    // Parse ALL logs to build complete state from scratch
    let blocks = [];
    let current = null;
    let requests = {};
    let blockMap = new Map(); // Track unique blocks
    let tokens = { total: { prompt: 0, completion: 0, total: 0 }, byBlock: {} };
    
    let flowTotalBlocks = 0;
    let flowCompletedBlocks = 0;
    let hasSeenBlockExecution = false; // Only consider progress after seeing block execution
    
    generationLogs.forEach(log => {
      const msg = stripAnsi(log.message || '');
      
      // Detect NEW run starting - reset everything when we see the start marker
      // This handles cases where rawOutput contains logs from a previous run
      // Check for various start markers that indicate a fresh run
      const isRunStart = msg.includes('🚀 Starting generation') || 
                         msg.includes('Starting generation...') ||
                         msg.includes('Starting generation') ||
                         msg.match(/^🚀.*generation/i);
      if (isRunStart) {
        // Reset all state for new run
        blocks = [];
        current = null;
        requests = {};
        blockMap = new Map();
        tokens = { total: { prompt: 0, completion: 0, total: 0 }, byBlock: {} };
        flowTotalBlocks = 0;
        flowCompletedBlocks = 0;
        hasSeenBlockExecution = false;
        totalBlocksRef.current = 0;
        return; // Skip this log entry
      }
      
      // Detect flow start with total block count - be more specific
      // Format: "Starting flow 'Name' v1.0.0 with X samples across Y blocks"
      const flowStartMatch = msg.match(/Starting flow.*?across (\d+) blocks/);
      if (flowStartMatch) {
        flowTotalBlocks = parseInt(flowStartMatch[1]);
      }
      
      // Detect block execution start (handles INFO prefix and line breaks)
      // Format: "Executing block 1/35: block_name (LLMChatBlock)" or similar
      const blockStartMatch = msg.match(/Executing block (\d+)\/(\d+):\s*([\w_]+)\s*\(?([\w]+)?\)?/) ||
                              msg.match(/block (\d+)\/(\d+).*?([\w_]+)/i);
      if (blockStartMatch) {
        hasSeenBlockExecution = true; // We've now seen actual block execution
        
        const blockName = blockStartMatch[3];
        const blockType = blockStartMatch[4] || 'LLMBlock';
        const blockNum = parseInt(blockStartMatch[1]);
        const detectedTotal = parseInt(blockStartMatch[2]);
        
        // Persist total blocks in ref (use the max seen to handle any parsing issues)
        if (detectedTotal > totalBlocksRef.current) {
          totalBlocksRef.current = detectedTotal;
        }
        
        // If we're executing block N, blocks 1 through N-1 are completed
        // Mark all previously added blocks as completed
        if (blockNum > 1) {
          blocks.forEach(b => {
            if (b.number < blockNum) {
              b.status = 'completed';
            }
          });
          flowCompletedBlocks = blockNum - 1;
        } else {
          flowCompletedBlocks = 0;
        }
        flowTotalBlocks = detectedTotal;
        
        current = { name: blockName, type: blockType, status: 'running', number: blockNum };
        if (!blockMap.has(blockName)) {
          blockMap.set(blockName, current);
          blocks.push(current);
        }
      }
      
      // Detect block completion - multiple patterns
      const blockCompleteMatch = msg.match(/Block '([\w_]+)' completed/) ||
                                  msg.match(/Block\s+([\w_]+)\s+completed/i);
      if (blockCompleteMatch) {
        const blockName = blockCompleteMatch[1];
        if (blockMap.has(blockName)) {
          blockMap.get(blockName).status = 'completed';
        }
      }
      
      // Detect block completion from tqdm 100% - indicates block finished
      const tqdm100Match = msg.match(/([\w_]+):\s*100%/);
      if (tqdm100Match) {
        const blockName = tqdm100Match[1];
        if (blockMap.has(blockName)) {
          blockMap.get(blockName).status = 'completed';
          flowCompletedBlocks = Math.max(flowCompletedBlocks, 
            Array.from(blockMap.values()).filter(b => b.status === 'completed').length);
        }
      }
      
      
      // Detect tqdm progress bars - multiple formats
      // New format from sdg_hub: block_name:  50%|█████| 5/10 [00:05<00:05, 1.00req/s]
      // Also handles: block_name:  50%|#####     | 5/10 [00:05<00:05, 1.00req/s]
      const tqdmMatch = msg.match(/([\w_]+):\s*(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/) ||
                        msg.match(/\[([\w_]+)\]\s+LLM Requests:\s+(\d+)%\|.*?\|\s*(\d+)\/(\d+)/) ||
                        msg.match(/([\w_]+):\s*(\d+)%.*?(\d+)\/(\d+)/);
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
        
        // If we see a tqdm for a block, that block is running
        if (!blockMap.has(blockName)) {
          current = { name: blockName, type: 'LLMChatBlock', status: 'running' };
          blockMap.set(blockName, current);
          blocks.push(current);
        }
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
    
    // Only calculate progress if we've seen actual block execution
    // This prevents showing stale data from previous runs
    let actualCompletedBlocks = 0;
    let finalTotalBlocks = 0;
    let finalProgress = 0;
    
    if (hasSeenBlockExecution) {
      // Use the total from the most recent "Executing block X/Y" message
      finalTotalBlocks = flowTotalBlocks > 0 ? flowTotalBlocks : totalBlocksRef.current;
      
      // Count completed blocks based on the last block number we saw executing
      // If we're on block N, then blocks 1 to N-1 are completed
      actualCompletedBlocks = flowCompletedBlocks;
      
      // Check if generation was cancelled/stopped by user
      const cancelled = generationLogs && generationLogs.some(log => {
        const msg = typeof log === 'string' ? log : (log?.message || '');
        return msg.includes('cancelled by user');
      });
      
      // If generation is NOT running and we've seen blocks, check if completed or stopped
      const isComplete = !isGenerating && hasSeenBlockExecution && finalTotalBlocks > 0 && !cancelled;
      
      if (isComplete) {
        actualCompletedBlocks = finalTotalBlocks;
        finalProgress = 100;
        current = null;
      } else if (finalTotalBlocks > 0) {
        // Calculate progress based on completed blocks
        finalProgress = (actualCompletedBlocks / finalTotalBlocks) * 100;
      }
      
      if (cancelled) {
        setWasCancelled(true);
      }
    }
    
    // Update state
    setBlockProgress(blocks);
    setCurrentBlock(current);
    setRequestStats(requests);
    setTotalBlocks(finalTotalBlocks);
    setCompletedBlocks(actualCompletedBlocks);
    setTokenStats(tokens);
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
                      {totalBlocks > 0 ? `${completedBlocks} / ${totalBlocks}` : 'Detecting...'}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Current Block</DescriptionListTerm>
                    <DescriptionListDescription>
                      {!isGenerating && wasCancelled ? (
                        <span style={{ color: '#c9190b', fontWeight: 'bold' }}>Stopped</span>
                      ) : !isGenerating && overallProgress >= 99 ? (
                        <span style={{ color: '#3e8635', fontWeight: 'bold' }}>✅ All Complete</span>
                      ) : currentBlock ? (
                        currentBlock.name
                      ) : isGenerating ? (
                        'Starting...'
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
                      ) : wasCancelled ? (
                        <Badge style={{ background: '#c9190b', color: 'white' }}>Stopped</Badge>
                      ) : overallProgress >= 99 ? (
                        <Badge style={{ background: '#3e8635', color: 'white' }}><CheckCircleIcon /> Completed</Badge>
                      ) : (
                        <Badge><CheckCircleIcon /> Idle</Badge>
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
              {overallProgress >= 99 && !isGenerating && !wasCancelled && (
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

