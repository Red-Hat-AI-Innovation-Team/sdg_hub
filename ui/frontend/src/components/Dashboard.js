import React, { useState, useEffect, useMemo } from 'react';
import {
  PageSection,
  Title,
  Text,
  Card,
  CardTitle,
  CardBody,
  Grid,
  GridItem,
  Button,
  Flex,
  FlexItem,
  Label,
  Spinner,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
  Tooltip,
  Badge,
  DataList,
  DataListItem,
  DataListItemRow,
  DataListItemCells,
  DataListCell,
  DataListAction,
  SearchInput,
  Modal,
  ModalVariant,
  Checkbox,
} from '@patternfly/react-core';
import {
  CubesIcon,
  PlusCircleIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  InProgressIcon,
  PauseCircleIcon,
  DownloadIcon,
  PlayIcon,
  ClockIcon,
  CogIcon,
  TrashIcon,
  DatabaseIcon,
  ChartBarIcon,
  TachometerAltIcon,
  OutlinedCommentsIcon,
  AngleDownIcon,
  AngleRightIcon,
  FolderIcon,
  FileIcon,
} from '@patternfly/react-icons';

// Collapsible Section Component
const CollapsibleSection = ({ title, icon: Icon, badge, isCollapsed, onToggle, headerActions, children }) => {
  return (
    <Card style={{ marginBottom: '1.5rem' }}>
      <CardTitle
        style={{
          cursor: 'pointer',
          userSelect: 'none',
          padding: isCollapsed ? '0.75rem 1rem' : '1rem 1.25rem',
          transition: 'all 0.2s ease',
        }}
        onClick={onToggle}
      >
        <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} alignItems={{ default: 'alignItemsCenter' }}>
          <FlexItem>
            <Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.75rem' }}>
              <FlexItem>
                {isCollapsed ? (
                  <AngleRightIcon style={{ fontSize: '1.25rem', color: '#6a6e73' }} />
                ) : (
                  <AngleDownIcon style={{ fontSize: '1.25rem', color: '#6a6e73' }} />
                )}
              </FlexItem>
              <FlexItem>
                <Title headingLevel="h2" size={isCollapsed ? 'md' : 'xl'} style={{ transition: 'font-size 0.2s ease' }}>
                  {Icon && <Icon style={{ marginRight: '0.75rem' }} />}
                  {title}
                  {badge !== undefined && badge !== null && (
                    <Badge style={{ marginLeft: '0.75rem' }} isRead>{badge}</Badge>
                  )}
                </Title>
              </FlexItem>
            </Flex>
          </FlexItem>
          {headerActions && !isCollapsed && (
            <FlexItem onClick={(e) => e.stopPropagation()}>
              {headerActions}
            </FlexItem>
          )}
        </Flex>
      </CardTitle>
      {!isCollapsed && (
        <CardBody style={{ padding: '1.25rem 1.5rem' }}>
          {children}
        </CardBody>
      )}
    </Card>
  );
};
import { savedConfigAPI, preprocessingAPI, runsAPI, executionAPI, customFlowsAPI } from '../services/api';

// Helper to strip ANSI codes
const stripAnsi = (str) => {
  if (!str) return '';
  return str.replace(/\x1b\[[0-9;]*m/g, '')
            .replace(/\x1b\]8;;[^\x07]*\x07[^\x1b]*\x1b\]8;;\x07/g, '')
            .replace(/\x1b\]8;;[^\x1b]*\x1b\\/g, '');
};

// Circular Progress Component
const CircularProgress = ({ percentage, size = 120, strokeWidth = 10, status, children }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;
  
  const getColor = () => {
    switch (status) {
      case 'running': return '#0066cc';
      case 'completed': return '#3e8635';
      case 'failed': return '#c9190b';
      case 'cancelled': return '#f0ab00';
      case 'configured': return '#6a6e73';
      default: return '#d2d2d2';
    }
  };
  
  const color = getColor();
  
  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e0e0e0"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.3s ease' }}
        />
      </svg>
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        textAlign: 'center',
      }}>
        {children}
      </div>
    </div>
  );
};

// Parse LLM requests from the generation logs
const parseLLMRequests = (rawOutput, debug = false) => {
  if (!rawOutput) return 0;
  
  const cleanOutput = stripAnsi(rawOutput);
  const lines = cleanOutput.split('\n');
  let totalRequests = 0;
  let llmBlocksFound = [];
  
  for (const line of lines) {
    // Method 1: Look for LLMChatBlock completion logs
    // Format: "llm_chat_block - INFO - Generation completed successfully for X samples"
    const llmCompletionMatch = line.match(/llm_chat_block.*Generation completed successfully for (\d+) samples/i);
    if (llmCompletionMatch) {
      const samples = parseInt(llmCompletionMatch[1], 10);
      if (!isNaN(samples)) {
        totalRequests += samples;
        llmBlocksFound.push({ type: 'completion_log', line: line.substring(0, 120), samples });
      }
      continue;
    }
    
    // Method 2: Look for summary table rows with LLMChatBlock (fallback)
    // Format: │ block_name │ LLMChatBlock │ duration │ rows_in → rows_out │
    if (line.includes('LLMChatBlock')) {
      const rowsMatch = line.match(/(\d+)\s*(?:→|->)\s*(\d+)/);
      if (rowsMatch) {
        const inputRows = parseInt(rowsMatch[1], 10);
        if (!isNaN(inputRows)) {
          totalRequests += inputRows;
          llmBlocksFound.push({ type: 'summary_table', line: line.substring(0, 120), inputRows });
        }
      }
    }
  }
  
  if (debug) {
    console.log('=== LLM Request Parsing Debug ===');
    console.log('Raw output length:', rawOutput.length);
    console.log('Clean output length:', cleanOutput.length);
    console.log('Total lines:', lines.length);
    console.log('LLM blocks found:', llmBlocksFound.length);
    console.log('LLM blocks:', llmBlocksFound);
    console.log('Total requests:', totalRequests);
    console.log('=================================');
  }
  
  return totalRequests;
};

// Parse block progress from rawOutput
const parseBlockProgress = (rawOutput, status) => {
  if (!rawOutput) {
    return { currentBlock: 0, totalBlocks: 0, currentBlockName: '', percentage: 0, samplesProcessed: 0, totalSamples: 0 };
  }
  
  const cleanOutput = stripAnsi(rawOutput);
  
  let totalBlocks = 0;
  let currentBlock = 0;
  let currentBlockName = '';
  let samplesProcessed = 0;
  let totalSamples = 0;
  
  const blockMatches = [...cleanOutput.matchAll(/Executing block (\d+)\/(\d+):\s*([\w_]+)/g)];
  for (const match of blockMatches) {
    currentBlock = parseInt(match[1]);
    totalBlocks = parseInt(match[2]);
    currentBlockName = match[3];
  }
  
  const tqdmMatches = [...cleanOutput.matchAll(/(\d+)\/(\d+)\s*(?:\[|it)/g)];
  if (tqdmMatches.length > 0) {
    const lastMatch = tqdmMatches[tqdmMatches.length - 1];
    samplesProcessed = parseInt(lastMatch[1]);
    totalSamples = parseInt(lastMatch[2]);
  }
  
  const percentMatches = [...cleanOutput.matchAll(/\[?\s*(\d+)%\s*\]?/g)];
  let lastPercent = 0;
  if (percentMatches.length > 0) {
    lastPercent = parseInt(percentMatches[percentMatches.length - 1][1]);
  }
  
  let percentage = 0;
  if (status === 'completed') {
    percentage = 100;
    if (totalBlocks > 0) currentBlock = totalBlocks;
    currentBlockName = 'Complete';
  } else if (totalBlocks > 0) {
    const completedBlocks = currentBlock > 0 ? currentBlock - 1 : 0;
    const blockProgress = (completedBlocks / totalBlocks) * 100;
    if (totalSamples > 0 && currentBlock > 0) {
      const sampleProgress = (samplesProcessed / totalSamples) * (100 / totalBlocks);
      percentage = Math.min(99, blockProgress + sampleProgress);
    } else {
      percentage = Math.min(99, blockProgress);
    }
  } else if (lastPercent > 0) {
    percentage = Math.min(99, lastPercent);
  }
  
  return {
    currentBlock,
    totalBlocks,
    currentBlockName,
    percentage: Math.round(percentage),
    samplesProcessed,
    totalSamples,
  };
};

// Status Badge Component
const StatusBadge = ({ status }) => {
  const config = {
    running: { color: 'blue', icon: <InProgressIcon />, label: 'Running' },
    completed: { color: 'green', icon: <CheckCircleIcon />, label: 'Completed' },
    failed: { color: 'red', icon: <ExclamationCircleIcon />, label: 'Failed' },
    cancelled: { color: 'orange', icon: <PauseCircleIcon />, label: 'Stopped' },
    configured: { color: 'grey', icon: <CogIcon />, label: 'Configured' },
    pending: { color: 'grey', icon: <ClockIcon />, label: 'Pending' },
  };
  
  const { color, icon, label } = config[status] || config.pending;
  
  return (
    <Label color={color} icon={icon} style={{ marginTop: '0.5rem' }}>
      {label}
    </Label>
  );
};

// Flow Card Component - Larger version
const FlowCard = ({ config, executionState, onNavigate, onDownload, onRun }) => {
  const configId = config.id;
  const configName = config.name || config.flow_name || 'Unnamed Flow';
  const status = executionState?.status;
  const isRunning = executionState?.isRunning;
  const rawOutput = executionState?.rawOutput;
  const result = executionState?.result;
  const runId = executionState?.runId;
  const startTime = executionState?.startTime;
  const completedAt = executionState?.completedAt;
  
  const progress = parseBlockProgress(rawOutput, status);
  const displayStatus = isRunning ? 'running' : (status || 'configured');
  
  const getDuration = () => {
    if (!startTime) return null;
    const start = new Date(startTime);
    const end = completedAt ? new Date(completedAt) : new Date();
    const seconds = Math.floor((end - start) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };
  
  const duration = getDuration();
  
  return (
    <Card style={{ height: '100%', position: 'relative', minHeight: '280px' }}>
      <CardBody style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <CircularProgress 
          percentage={displayStatus === 'configured' ? 0 : progress.percentage} 
          status={displayStatus}
          size={100}
          strokeWidth={8}
        >
          <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: displayStatus === 'running' ? '#0066cc' : '#151515' }}>
            {displayStatus === 'configured' ? '—' : `${progress.percentage}%`}
          </div>
          {displayStatus === 'running' && progress.totalBlocks > 0 && (
            <div style={{ fontSize: '0.7rem', color: '#6a6e73' }}>
              {progress.currentBlock}/{progress.totalBlocks}
            </div>
          )}
        </CircularProgress>
        
        <Text style={{ marginTop: '1rem', textAlign: 'center', fontSize: '0.95rem', fontWeight: 600, lineHeight: 1.3 }}>
          {configName.length > 50 ? configName.substring(0, 50) + '...' : configName}
        </Text>
        
        <div style={{ marginTop: '0.75rem' }}>
          <StatusBadge status={displayStatus} />
        </div>
        
        {(duration || (displayStatus === 'completed' && result?.num_samples)) && (
          <Text style={{ fontSize: '0.85rem', color: '#6a6e73', marginTop: '0.5rem' }}>
            {duration && <><ClockIcon style={{ marginRight: '0.25rem', fontSize: '0.8rem' }} />{duration}</>}
            {displayStatus === 'completed' && result?.num_samples && (
              <span style={{ color: '#3e8635', fontWeight: 500 }}> • {result.num_samples} samples</span>
            )}
          </Text>
        )}
        
        <Flex style={{ marginTop: 'auto', paddingTop: '1rem', width: '100%' }} justifyContent={{ default: 'justifyContentCenter' }}>
          {displayStatus === 'completed' && runId && (
            <FlexItem>
              <Button 
                variant="primary" 
                icon={<DownloadIcon />}
                onClick={() => onDownload(runId)}
                style={{ fontSize: '0.9rem', padding: '0.5rem 1rem' }}
              >
                Download Generated Dataset
              </Button>
            </FlexItem>
          )}
          {displayStatus === 'configured' && (
            <FlexItem>
              <Button 
                variant="primary" 
                icon={<PlayIcon />}
                onClick={() => onRun(config)}
                style={{ fontSize: '0.9rem', padding: '0.5rem 1rem' }}
              >
                Run
              </Button>
            </FlexItem>
          )}
          {displayStatus === 'running' && (
            <FlexItem>
              <Button 
                variant="secondary" 
                onClick={() => onNavigate('flows', { viewConfig: configId })}
                style={{ fontSize: '0.9rem', padding: '0.5rem 1rem' }}
              >
                View Progress
              </Button>
            </FlexItem>
          )}
        </Flex>
      </CardBody>
    </Card>
  );
};

const Dashboard = ({ executionStates, onUpdateExecutionState, onNavigate }) => {
  const [configurations, setConfigurations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [datasets, setDatasets] = useState([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);
  const [datasetSearchValue, setDatasetSearchValue] = useState('');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [runHistory, setRunHistory] = useState([]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [selectedStatsFlows, setSelectedStatsFlows] = useState([]);
  const [statsFlowSearchValue, setStatsFlowSearchValue] = useState('');
  const [selectedMetrics, setSelectedMetrics] = useState(['lastsamples', 'lastduration', 'llmrequests', 'runs']);
  
  // Custom flows state
  const [customFlows, setCustomFlows] = useState([]);
  const [isLoadingCustomFlows, setIsLoadingCustomFlows] = useState(true);
  const [customFlowSearchValue, setCustomFlowSearchValue] = useState('');
  const [customFlowToDelete, setCustomFlowToDelete] = useState(null);
  const [deleteCustomFlowModalOpen, setDeleteCustomFlowModalOpen] = useState(false);
  const [isDeletingCustomFlow, setIsDeletingCustomFlow] = useState(false);
  const [expandedCustomFlows, setExpandedCustomFlows] = useState({});
  const [deleteAllCustomFlowsModalOpen, setDeleteAllCustomFlowsModalOpen] = useState(false);
  const [isDeletingAllCustomFlows, setIsDeletingAllCustomFlows] = useState(false);
  
  // Collapsible section states - only "Your Flows" is expanded by default
  const [collapsedSections, setCollapsedSections] = useState({
    flows: false,        // Your Flows - expanded by default
    statistics: true,    // Flow Statistics - collapsed by default
    datasets: true,      // Preprocessed Datasets - collapsed by default
    customFlows: true,   // My Custom Flows - collapsed by default
  });
  
  const toggleSection = (section) => {
    setCollapsedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };
  
  const activeEventSourcesRef = React.useRef(new Map());
  
  React.useEffect(() => {
    return () => {
      activeEventSourcesRef.current.forEach((eventSource) => {
        eventSource.close();
      });
      activeEventSourcesRef.current.clear();
    };
  }, []);
  
  useEffect(() => {
    const loadConfigurations = async () => {
      try {
        setIsLoading(true);
        const response = await savedConfigAPI.list();
        setConfigurations(response.configurations || []);
      } catch (error) {
        console.error('Failed to load configurations:', error);
        setConfigurations([]);
      } finally {
        setIsLoading(false);
      }
    };
    loadConfigurations();
  }, []);
  
  useEffect(() => {
    loadDatasets();
  }, []);
  
  const loadDatasets = async () => {
    try {
      setIsLoadingDatasets(true);
      const response = await preprocessingAPI.listDatasets();
      setDatasets(response.datasets || []);
    } catch (error) {
      console.error('Failed to load datasets:', error);
      setDatasets([]);
    } finally {
      setIsLoadingDatasets(false);
    }
  };
  
  // Function to load run history - accessible throughout the component
  const loadRunHistory = async () => {
    try {
      setIsLoadingRuns(true);
      const response = await runsAPI.list();
      setRunHistory(response.runs || []);
    } catch (error) {
      console.error('Failed to load run history:', error);
      setRunHistory([]);
    } finally {
      setIsLoadingRuns(false);
    }
  };
  
  useEffect(() => {
    loadRunHistory();
  }, []);
  
  // Load custom flows
  const loadCustomFlows = async () => {
    try {
      setIsLoadingCustomFlows(true);
      const response = await customFlowsAPI.list();
      setCustomFlows(response.flows || []);
    } catch (error) {
      console.error('Failed to load custom flows:', error);
      setCustomFlows([]);
    } finally {
      setIsLoadingCustomFlows(false);
    }
  };
  
  useEffect(() => {
    loadCustomFlows();
  }, []);
  
  // Filter custom flows by search
  const filteredCustomFlows = useMemo(() => {
    if (!customFlowSearchValue.trim()) return customFlows;
    const searchLower = customFlowSearchValue.toLowerCase();
    return customFlows.filter(flow => 
      flow.name?.toLowerCase().includes(searchLower) ||
      flow.description?.toLowerCase().includes(searchLower) ||
      flow.directory_name?.toLowerCase().includes(searchLower)
    );
  }, [customFlows, customFlowSearchValue]);
  
  // Handle custom flow file download
  const handleDownloadCustomFlowFile = (flowName, filename) => {
    const url = customFlowsAPI.getFileDownloadUrl(flowName, filename);
    window.open(url, '_blank');
  };
  
  // Handle custom flow ZIP download
  const handleDownloadCustomFlowZip = (flowName) => {
    const url = customFlowsAPI.getZipDownloadUrl(flowName);
    window.open(url, '_blank');
  };
  
  // Handle custom flow delete
  const handleDeleteCustomFlow = async () => {
    if (!customFlowToDelete) return;
    try {
      setIsDeletingCustomFlow(true);
      await customFlowsAPI.delete(customFlowToDelete.directory_name);
      setCustomFlows(prev => prev.filter(f => f.directory_name !== customFlowToDelete.directory_name));
      setDeleteCustomFlowModalOpen(false);
      setCustomFlowToDelete(null);
    } catch (error) {
      console.error('Failed to delete custom flow:', error);
      alert('Failed to delete custom flow: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsDeletingCustomFlow(false);
    }
  };
  
  // Handle delete all custom flows
  const handleDeleteAllCustomFlows = async () => {
    try {
      setIsDeletingAllCustomFlows(true);
      const result = await customFlowsAPI.deleteAll();
      setCustomFlows([]);
      setDeleteAllCustomFlowsModalOpen(false);
      if (result.status === 'partial') {
        alert(`Deleted ${result.deleted_count} flows. Some errors occurred: ${result.errors?.join(', ')}`);
      }
    } catch (error) {
      console.error('Failed to delete all custom flows:', error);
      alert('Failed to delete all custom flows: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsDeletingAllCustomFlows(false);
    }
  };
  
  // Toggle expanded state for a custom flow
  const toggleCustomFlowExpanded = (flowName) => {
    setExpandedCustomFlows(prev => ({
      ...prev,
      [flowName]: !prev[flowName],
    }));
  };
  
  const stats = useMemo(() => {
    let running = 0;
    let completed = 0;
    let failed = 0;
    
    configurations.forEach(config => {
      const state = executionStates?.[config.id];
      if (state?.isRunning || state?.status === 'running') {
        running++;
      } else if (state?.status === 'completed') {
        completed++;
      } else if (state?.status === 'failed') {
        failed++;
      }
    });
    
    return { running, completed, failed, total: configurations.length };
  }, [configurations, executionStates]);
  
  const sortedConfigurations = useMemo(() => {
    return [...configurations].sort((a, b) => {
      const stateA = executionStates?.[a.id];
      const stateB = executionStates?.[b.id];
      
      if ((stateA?.isRunning || stateA?.status === 'running') && 
          !(stateB?.isRunning || stateB?.status === 'running')) return -1;
      if (!(stateA?.isRunning || stateA?.status === 'running') && 
          (stateB?.isRunning || stateB?.status === 'running')) return 1;
      
      const toTimestamp = (val) => {
        if (typeof val === 'number') return val;
        if (typeof val === 'string') return new Date(val).getTime();
        return 0;
      };
      const aTime = toTimestamp(stateA?.completedAt || stateA?.startTime) || new Date(a.created_at || 0).getTime();
      const bTime = toTimestamp(stateB?.completedAt || stateB?.startTime) || new Date(b.created_at || 0).getTime();
      return bTime - aTime;
    });
  }, [configurations, executionStates]);
  
  const flowStatistics = useMemo(() => {
    if (!runHistory.length || !configurations.length) {
      return { byFlow: [], allFlowsWithRuns: [], maxLastSamples: 1, maxLastDuration: 1, maxRuns: 1 };
    }
    
    const flowStats = {};
    
    configurations.forEach(config => {
      flowStats[config.id] = {
        configId: config.id,
        flowName: config.name || config.flow_name || 'Unnamed Flow',
        lastRunSamples: 0,
        lastRunDuration: 0,
        lastRunLLMRequests: 0,
        lastRunTime: null,
        runCount: 0,
      };
    });
    
    // Sort runs by start_time descending to find last run for each config
    const sortedRuns = [...runHistory].sort((a, b) => {
      const timeA = new Date(a.start_time || 0).getTime();
      const timeB = new Date(b.start_time || 0).getTime();
      return timeB - timeA;
    });
    
    sortedRuns.forEach(run => {
      if (run.config_id && flowStats[run.config_id]) {
        flowStats[run.config_id].runCount++;
        // Only set last run stats if this is the most recent completed run
        if (run.status === 'completed' && !flowStats[run.config_id].lastRunTime) {
          flowStats[run.config_id].lastRunSamples = run.output_samples || 0;
          flowStats[run.config_id].lastRunDuration = run.duration_seconds || 0;
          // Get LLM stats from llm_statistics (detailed) or fallback to llm_requests field
          const llmStats = run.llm_statistics;
          flowStats[run.config_id].lastRunLLMRequests = llmStats?.total_llm_requests || run.llm_requests || 0;
          flowStats[run.config_id].lastRunTime = run.start_time;
        }
      }
    });
    
    const allFlowsWithRuns = Object.values(flowStats)
      .filter(stat => stat.runCount > 0)
      .sort((a, b) => b.lastRunSamples - a.lastRunSamples);
    
    const byFlow = selectedStatsFlows.length > 0
      ? allFlowsWithRuns.filter(stat => selectedStatsFlows.includes(stat.configId))
      : allFlowsWithRuns;
    
    return {
      byFlow,
      allFlowsWithRuns,
      maxLastSamples: byFlow.length > 0 ? Math.max(...byFlow.map(s => s.lastRunSamples || 0), 1) : 1,
      maxLastDuration: byFlow.length > 0 ? Math.max(...byFlow.map(s => s.lastRunDuration || 0), 1) : 1,
      maxLastLLMRequests: byFlow.length > 0 ? Math.max(...byFlow.map(s => s.lastRunLLMRequests || 0), 1) : 1,
      maxRuns: byFlow.length > 0 ? Math.max(...byFlow.map(s => s.runCount || 0), 1) : 1,
    };
  }, [runHistory, configurations, selectedStatsFlows]);
  
  const filteredStatsFlows = useMemo(() => {
    if (!statsFlowSearchValue.trim()) {
      return flowStatistics.allFlowsWithRuns;
    }
    const searchLower = statsFlowSearchValue.toLowerCase();
    return flowStatistics.allFlowsWithRuns.filter(stat => 
      stat.flowName.toLowerCase().includes(searchLower)
    );
  }, [flowStatistics.allFlowsWithRuns, statsFlowSearchValue]);
  
  const handleDownload = async (runId) => {
    try {
      await runsAPI.download(runId);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };
  
  const handleRunGeneration = async (config) => {
    const configId = config.id;
    const runId = `run_${configId}_${Date.now()}`;
    const startTime = new Date().toISOString();
    
    try {
      const modelConfig = config.model_configuration || config.model_config || {};
      const datasetConfig = config.dataset_configuration || config.dataset_config || {};
      
      const runRecord = {
        run_id: runId,
        config_id: configId,
        flow_name: config.flow_name,
        flow_type: config.flow_id?.startsWith('custom-') ? 'custom' : 'existing',
        model_name: modelConfig.model || 'Unknown',
        status: 'running',
        start_time: startTime,
        input_samples: datasetConfig.num_samples || 0,
        dataset_file: datasetConfig.data_files || null,
      };
      
      await runsAPI.create(runRecord);
      
      onUpdateExecutionState(configId, {
        configId: configId,
        configName: config.name || config.flow_name,
        flowName: config.flow_name,
        type: 'generate',
        isRunning: true,
        status: 'running',
        rawOutput: '🚀 Starting generation...\n',
        result: null,
        runId: runId,
        startTime: startTime,
        isRestarting: true,
      });
      
      await savedConfigAPI.load(configId);
      
      onUpdateExecutionState(configId, prev => ({
        ...prev,
        rawOutput: (prev?.rawOutput || '') + '✅ Configuration loaded\n',
      }));
      
      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const streamUrl = `${API_BASE_URL}/api/flow/generate-stream?max_concurrency=100`;
      
      const eventSource = new EventSource(streamUrl);
      activeEventSourcesRef.current.set(configId, eventSource);
      
      // Track accumulated raw output locally to avoid stale closure issues
      let accumulatedRawOutput = '🚀 Starting generation...\n✅ Configuration loaded\n';
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'log' || data.type === 'start') {
            accumulatedRawOutput += data.message + '\n';
            onUpdateExecutionState(configId, prev => ({
              ...prev,
              rawOutput: (prev?.rawOutput || '') + data.message + '\n',
            }));
          } else if (data.type === 'complete') {
            eventSource.close();
            activeEventSourcesRef.current.delete(configId);
            
            const endTime = new Date().toISOString();
            const duration = (new Date(endTime) - new Date(startTime)) / 1000;
            
            // Update run record with completion status AND llm_requests from backend
            // The backend tracks llm_requests during generation and sends it in the complete event
            console.log('Generation complete. LLM requests from backend:', data.llm_requests);
            runsAPI.update(runId, {
              status: 'completed',
              end_time: endTime,
              duration_seconds: duration,
              output_samples: data.num_samples,
              output_columns: data.num_columns,
              output_file: data.output_file ? `outputs/${data.output_file}` : null,
              llm_requests: data.llm_requests || 0,  // Save LLM requests from backend
            }).then(() => {
              // Refresh run history to show updated stats
              loadRunHistory();
            }).catch(err => console.error('Failed to update run record:', err));
            
            onUpdateExecutionState(configId, prev => ({
              ...prev,
              isRunning: false,
              status: 'completed',
              rawOutput: (prev?.rawOutput || '') + `\n✅ Generation completed! ${data.num_samples} samples, ${data.num_columns} columns\n`,
              result: data,
              outputSamples: data.num_samples,
              completedAt: Date.now(),
            }));
          } else if (data.type === 'error') {
            eventSource.close();
            activeEventSourcesRef.current.delete(configId);
            
            runsAPI.update(runId, {
              status: 'failed',
              error_message: data.message,
            }).catch(err => console.error('Failed to update run record:', err));
            
            onUpdateExecutionState(configId, prev => ({
              ...prev,
              isRunning: false,
              status: 'failed',
              error: data.message,
              rawOutput: (prev?.rawOutput || '') + `\n❌ Error: ${data.message}\n`,
            }));
          }
        } catch (err) {
          console.error('Error parsing SSE message:', err);
        }
      };
      
      eventSource.onerror = () => {
        eventSource.close();
        activeEventSourcesRef.current.delete(configId);
        
        // Update backend run record so it doesn't stay stuck as "running"
        runsAPI.update(runId, {
          status: 'failed',
          error_message: 'Connection lost',
        }).catch(err => console.error('Failed to update run record on connection loss:', err));
        
        onUpdateExecutionState(configId, prev => ({
          ...prev,
          isRunning: false,
          status: 'failed',
          error: 'Connection lost',
          rawOutput: (prev?.rawOutput || '') + '\n❌ Connection lost\n',
        }));
      };
      
    } catch (error) {
      console.error('Failed to start generation:', error);
      
      // Update backend run record if it was already created
      runsAPI.update(runId, {
        status: 'failed',
        error_message: error.message,
      }).catch(err => console.error('Failed to update run record on error:', err));
      
      onUpdateExecutionState(configId, {
        configId: configId,
        isRunning: false,
        status: 'failed',
        error: error.message,
        rawOutput: `❌ Error: ${error.message}\n`,
      });
    }
  };
  
  const filteredDatasets = useMemo(() => {
    if (!datasetSearchValue.trim()) {
      return datasets;
    }
    const searchLower = datasetSearchValue.toLowerCase();
    return datasets.filter(dataset => 
      dataset.name?.toLowerCase().includes(searchLower) ||
      dataset.source_file?.toLowerCase().includes(searchLower)
    );
  }, [datasets, datasetSearchValue]);
  
  const handleDownloadDataset = async (jobId) => {
    try {
      const url = preprocessingAPI.getDatasetDownloadUrl(jobId);
      window.open(url, '_blank');
    } catch (error) {
      console.error('Download failed:', error);
    }
  };
  
  const handleDeleteDataset = async () => {
    if (!datasetToDelete) return;
    
    try {
      setIsDeleting(true);
      await preprocessingAPI.deleteDataset(datasetToDelete.job_id);
      await loadDatasets();
      setDeleteModalOpen(false);
      setDatasetToDelete(null);
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <PageSection>
      {/* Your Flows Section */}
      <CollapsibleSection
        title="Your Flows"
        icon={CubesIcon}
        badge={configurations.length > 0 ? configurations.length : null}
        isCollapsed={collapsedSections.flows}
        onToggle={() => toggleSection('flows')}
        headerActions={
          <Button variant="link" onClick={() => onNavigate('flows')}>
            Manage All Flows →
          </Button>
        }
      >
        {/* Stats Overview - Larger */}
        <Flex style={{ marginBottom: '1.5rem', gap: '1.5rem', flexWrap: 'wrap' }}>
          <FlexItem>
            <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '0.75rem 1.25rem', border: '1px solid #d2d2d2', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <InProgressIcon style={{ fontSize: '1.5rem', color: '#0066cc' }} />
              <Text style={{ fontWeight: 700, fontSize: '1.25rem' }}>{stats.running}</Text>
              <Text style={{ color: '#6a6e73', fontSize: '1rem' }}>Running</Text>
            </div>
          </FlexItem>
          <FlexItem>
            <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '0.75rem 1.25rem', border: '1px solid #d2d2d2', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <CheckCircleIcon style={{ fontSize: '1.5rem', color: '#3e8635' }} />
              <Text style={{ fontWeight: 700, fontSize: '1.25rem' }}>{stats.completed}</Text>
              <Text style={{ color: '#6a6e73', fontSize: '1rem' }}>Completed</Text>
            </div>
          </FlexItem>
          <FlexItem>
            <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '0.75rem 1.25rem', border: '1px solid #d2d2d2', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <ExclamationCircleIcon style={{ fontSize: '1.5rem', color: '#c9190b' }} />
              <Text style={{ fontWeight: 700, fontSize: '1.25rem' }}>{stats.failed}</Text>
              <Text style={{ color: '#6a6e73', fontSize: '1rem' }}>Failed</Text>
            </div>
          </FlexItem>
          <FlexItem>
            <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '0.75rem 1.25rem', border: '1px solid #d2d2d2', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <CogIcon style={{ fontSize: '1.5rem', color: '#6a6e73' }} />
              <Text style={{ fontWeight: 700, fontSize: '1.25rem' }}>{stats.total}</Text>
              <Text style={{ color: '#6a6e73', fontSize: '1rem' }}>Configured</Text>
            </div>
          </FlexItem>
        </Flex>
        
        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '4rem' }}>
            <Spinner size="xl" />
            <Text style={{ marginTop: '1.5rem', fontSize: '1.1rem' }}>Loading configurations...</Text>
          </div>
        ) : (
          <Grid hasGutter style={{ gap: '1.5rem' }}>
            {sortedConfigurations.slice(0, 7).map((config) => (
              <GridItem key={config.id} span={3}>
                <FlowCard 
                  config={config}
                  executionState={executionStates?.[config.id]}
                  onNavigate={onNavigate}
                  onDownload={handleDownload}
                  onRun={handleRunGeneration}
                />
              </GridItem>
            ))}
            <GridItem span={3}>
              <Card 
                isSelectable 
                isClickable
                onClick={() => onNavigate('configure-flow', {})}
                style={{ height: '100%', minHeight: '280px', cursor: 'pointer', border: '2px dashed #d2d2d2', backgroundColor: '#fafafa' }}
              >
                <CardBody style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1.5rem' }}>
                  <div style={{ width: '80px', height: '80px', borderRadius: '50%', backgroundColor: '#e7f1fa', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                    <PlusCircleIcon style={{ fontSize: '2.5rem', color: '#0066cc' }} />
                  </div>
                  <Text style={{ textAlign: 'center', color: '#0066cc', fontSize: '1.1rem', fontWeight: 600 }}>
                    Add New Configuration
                  </Text>
                  <Text style={{ fontSize: '0.9rem', color: '#6a6e73', textAlign: 'center', marginTop: '0.5rem' }}>
                    Create a new data generation flow
                  </Text>
                </CardBody>
              </Card>
            </GridItem>
            {sortedConfigurations.length > 7 && (
              <GridItem span={12} style={{ textAlign: 'center', paddingTop: '1rem' }}>
                <Button variant="link" onClick={() => onNavigate('flows')} style={{ fontSize: '1rem' }}>
                  View all {sortedConfigurations.length} flows →
                </Button>
              </GridItem>
            )}
          </Grid>
        )}
      </CollapsibleSection>

      {/* Flow Statistics Section */}
      {flowStatistics.allFlowsWithRuns.length > 0 && (
        <CollapsibleSection
          title="Flow Statistics"
          icon={ChartBarIcon}
          isCollapsed={collapsedSections.statistics}
          onToggle={() => toggleSection('statistics')}
        >
          {isLoadingRuns ? (
            <div style={{ textAlign: 'center', padding: '3rem' }}>
              <Spinner size="xl" />
            </div>
          ) : (
            <Grid hasGutter>
              <GridItem span={4}>
                <Card isFlat style={{ height: '100%' }}>
                  <CardBody style={{ padding: '1rem' }}>
                    <Text style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>1. Select Flows</Text>
                    <SearchInput
                      placeholder="Search flows..."
                      value={statsFlowSearchValue}
                      onChange={(_, value) => setStatsFlowSearchValue(value)}
                      onClear={() => setStatsFlowSearchValue('')}
                      style={{ marginBottom: '0.75rem' }}
                    />
                    <Flex style={{ marginBottom: '0.5rem', gap: '0.75rem' }}>
                      <FlexItem>
                        <Button variant="link" isSmall onClick={() => setSelectedStatsFlows(filteredStatsFlows.map(s => s.configId))} isDisabled={filteredStatsFlows.length === 0} style={{ padding: 0, fontSize: '0.9rem' }}>Select All</Button>
                      </FlexItem>
                      <FlexItem>
                        <Button variant="link" isSmall onClick={() => setSelectedStatsFlows([])} isDisabled={selectedStatsFlows.length === 0} style={{ padding: 0, fontSize: '0.9rem' }}>Clear</Button>
                      </FlexItem>
                      {selectedStatsFlows.length > 0 && <FlexItem><Badge isRead style={{ fontSize: '0.85rem' }}>{selectedStatsFlows.length}</Badge></FlexItem>}
                    </Flex>
                    <div style={{ maxHeight: '220px', overflowY: 'auto', border: '1px solid #d2d2d2', borderRadius: '6px', padding: '0.5rem 0.75rem', marginBottom: '1.25rem' }}>
                      {filteredStatsFlows.length === 0 ? (
                        <Text style={{ color: '#6a6e73', fontSize: '0.95rem', textAlign: 'center', padding: '1rem' }}>No flows match</Text>
                      ) : (
                        filteredStatsFlows.map((stat) => (
                          <div key={stat.configId} style={{ padding: '0.5rem 0', borderBottom: '1px solid #f0f0f0' }}>
                            <Checkbox
                              id={`stats-flow-${stat.configId}`}
                              isChecked={selectedStatsFlows.includes(stat.configId)}
                              onChange={() => {
                                if (selectedStatsFlows.includes(stat.configId)) {
                                  setSelectedStatsFlows(selectedStatsFlows.filter(id => id !== stat.configId));
                                } else {
                                  setSelectedStatsFlows([...selectedStatsFlows, stat.configId]);
                                }
                              }}
                              label={<div style={{ fontSize: '0.9rem' }}><div style={{ fontWeight: 500, wordBreak: 'break-word' }}>{stat.flowName}</div><div style={{ fontSize: '0.8rem', color: '#6a6e73' }}>{stat.runCount || 0} runs • {(stat.lastRunSamples || 0).toLocaleString()} samples</div></div>}
                            />
                          </div>
                        ))
                      )}
                    </div>
                    <Text style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>2. Select Metrics</Text>
                    <div style={{ border: '1px solid #d2d2d2', borderRadius: '6px', padding: '0.75rem' }}>
                      <Checkbox id="metric-lastsamples" isChecked={selectedMetrics.includes('lastsamples')} onChange={() => selectedMetrics.includes('lastsamples') ? setSelectedMetrics(selectedMetrics.filter(m => m !== 'lastsamples')) : setSelectedMetrics([...selectedMetrics, 'lastsamples'])} label={<Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.5rem' }}><DatabaseIcon style={{ color: '#0066cc', fontSize: '1.1rem' }} /><span style={{ fontSize: '0.95rem' }}>Last Run Generated Samples</span></Flex>} style={{ marginBottom: '0.5rem' }} />
                      <Checkbox id="metric-lastduration" isChecked={selectedMetrics.includes('lastduration')} onChange={() => selectedMetrics.includes('lastduration') ? setSelectedMetrics(selectedMetrics.filter(m => m !== 'lastduration')) : setSelectedMetrics([...selectedMetrics, 'lastduration'])} label={<Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.5rem' }}><ClockIcon style={{ color: '#3e8635', fontSize: '1.1rem' }} /><span style={{ fontSize: '0.95rem' }}>Generation Time of Last Run</span></Flex>} style={{ marginBottom: '0.5rem' }} />
                      <Checkbox id="metric-llmrequests" isChecked={selectedMetrics.includes('llmrequests')} onChange={() => selectedMetrics.includes('llmrequests') ? setSelectedMetrics(selectedMetrics.filter(m => m !== 'llmrequests')) : setSelectedMetrics([...selectedMetrics, 'llmrequests'])} label={<Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.5rem' }}><OutlinedCommentsIcon style={{ color: '#6753ac', fontSize: '1.1rem' }} /><span style={{ fontSize: '0.95rem' }}>LLM Requests (Last Run)</span></Flex>} style={{ marginBottom: '0.5rem' }} />
                      <Checkbox id="metric-runs" isChecked={selectedMetrics.includes('runs')} onChange={() => selectedMetrics.includes('runs') ? setSelectedMetrics(selectedMetrics.filter(m => m !== 'runs')) : setSelectedMetrics([...selectedMetrics, 'runs'])} label={<Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.5rem' }}><TachometerAltIcon style={{ color: '#8a8d90', fontSize: '1.1rem' }} /><span style={{ fontSize: '0.95rem' }}>Number of Runs</span></Flex>} />
                    </div>
                  </CardBody>
                </Card>
              </GridItem>
              <GridItem span={8}>
                {selectedStatsFlows.length === 0 || selectedMetrics.length === 0 ? (
                  <Card isFlat style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <CardBody>
                      <EmptyState variant="sm">
                        <EmptyStateIcon icon={ChartBarIcon} />
                        <Title headingLevel="h4" size="lg">{selectedStatsFlows.length === 0 && selectedMetrics.length === 0 ? 'Select Flows and Metrics' : selectedStatsFlows.length === 0 ? 'Select Flows' : 'Select Metrics'}</Title>
                        <EmptyStateBody style={{ fontSize: '1rem' }}>{selectedStatsFlows.length === 0 && selectedMetrics.length === 0 ? 'Choose flows and metrics from the left panel to view charts.' : selectedStatsFlows.length === 0 ? 'Choose one or more flows to compare.' : 'Choose one or more metrics to display.'}</EmptyStateBody>
                      </EmptyState>
                    </CardBody>
                  </Card>
                ) : (
                  <Grid hasGutter>
                    {selectedMetrics.includes('lastsamples') && (
                      <GridItem span={selectedMetrics.length === 1 ? 12 : 6}>
                        <Card isFlat style={{ height: '100%' }}>
                          <CardTitle style={{ fontSize: '1rem', paddingBottom: '0.5rem' }}><DatabaseIcon style={{ marginRight: '0.5rem', color: '#0066cc' }} />Last Run Generated Samples</CardTitle>
                          <CardBody style={{ paddingTop: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', height: '180px', paddingBottom: '0.75rem', borderBottom: '1px solid #d2d2d2' }}>
                              {flowStatistics.byFlow.map((stat) => (
                                <Tooltip key={stat.configId} content={`${stat.flowName}: ${(stat.lastRunSamples || 0).toLocaleString()} samples`}>
                                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, maxWidth: '80px' }}>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.3rem', color: '#151515' }}>{(stat.lastRunSamples || 0) >= 1000 ? `${((stat.lastRunSamples || 0)/1000).toFixed(1)}k` : (stat.lastRunSamples || 0)}</div>
                                    <div style={{ width: '36px', height: `${Math.max(((stat.lastRunSamples || 0) / flowStatistics.maxLastSamples) * 130, 6)}px`, backgroundColor: '#0066cc', borderRadius: '4px 4px 0 0', transition: 'height 0.3s ease' }} />
                                  </div>
                                </Tooltip>
                              ))}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '0.5rem' }}>
                              {flowStatistics.byFlow.map((stat, idx) => (<Tooltip key={stat.configId} content={stat.flowName}><div style={{ flex: 1, maxWidth: '80px', textAlign: 'center', fontSize: '0.8rem', color: '#6a6e73' }}>Flow {idx + 1}</div></Tooltip>))}
                            </div>
                          </CardBody>
                        </Card>
                      </GridItem>
                    )}
                    {selectedMetrics.includes('lastduration') && (
                      <GridItem span={selectedMetrics.length === 1 ? 12 : 6}>
                        <Card isFlat style={{ height: '100%' }}>
                          <CardTitle style={{ fontSize: '1rem', paddingBottom: '0.5rem' }}><ClockIcon style={{ marginRight: '0.5rem', color: '#3e8635' }} />Generation Time of Last Run</CardTitle>
                          <CardBody style={{ paddingTop: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', height: '180px', paddingBottom: '0.75rem', borderBottom: '1px solid #d2d2d2' }}>
                              {flowStatistics.byFlow.map((stat) => {
                                const fmt = (s) => { if (!s) return '0s'; if (s < 60) return `${Math.round(s)}s`; const m = Math.floor(s / 60); if (m < 60) return `${m}m`; return `${Math.floor(m / 60)}h`; };
                                return (<Tooltip key={stat.configId} content={`${stat.flowName}: ${fmt(stat.lastRunDuration)}`}><div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, maxWidth: '80px' }}><div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.3rem', color: '#151515' }}>{fmt(stat.lastRunDuration)}</div><div style={{ width: '36px', height: `${Math.max((stat.lastRunDuration / flowStatistics.maxLastDuration) * 130, 6)}px`, backgroundColor: '#3e8635', borderRadius: '4px 4px 0 0', transition: 'height 0.3s ease' }} /></div></Tooltip>);
                              })}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '0.5rem' }}>
                              {flowStatistics.byFlow.map((stat, idx) => (<Tooltip key={stat.configId} content={stat.flowName}><div style={{ flex: 1, maxWidth: '80px', textAlign: 'center', fontSize: '0.8rem', color: '#6a6e73' }}>Flow {idx + 1}</div></Tooltip>))}
                            </div>
                          </CardBody>
                        </Card>
                      </GridItem>
                    )}
                    {selectedMetrics.includes('llmrequests') && (
                      <GridItem span={selectedMetrics.length === 1 ? 12 : 6}>
                        <Card isFlat style={{ height: '100%' }}>
                          <CardTitle style={{ fontSize: '1rem', paddingBottom: '0.5rem' }}><OutlinedCommentsIcon style={{ marginRight: '0.5rem', color: '#6753ac' }} />LLM Requests (Last Run)</CardTitle>
                          <CardBody style={{ paddingTop: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', height: '180px', paddingBottom: '0.75rem', borderBottom: '1px solid #d2d2d2' }}>
                              {flowStatistics.byFlow.map((stat) => (
                                <Tooltip key={stat.configId} content={`${stat.flowName}: ${(stat.lastRunLLMRequests || 0).toLocaleString()} LLM requests`}>
                                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, maxWidth: '80px' }}>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.3rem', color: '#151515' }}>{(stat.lastRunLLMRequests || 0) >= 1000 ? `${((stat.lastRunLLMRequests || 0)/1000).toFixed(1)}k` : (stat.lastRunLLMRequests || 0)}</div>
                                    <div style={{ width: '36px', height: `${Math.max(((stat.lastRunLLMRequests || 0) / flowStatistics.maxLastLLMRequests) * 130, 6)}px`, backgroundColor: '#6753ac', borderRadius: '4px 4px 0 0', transition: 'height 0.3s ease' }} />
                                  </div>
                                </Tooltip>
                              ))}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '0.5rem' }}>
                              {flowStatistics.byFlow.map((stat, idx) => (<Tooltip key={stat.configId} content={stat.flowName}><div style={{ flex: 1, maxWidth: '80px', textAlign: 'center', fontSize: '0.8rem', color: '#6a6e73' }}>Flow {idx + 1}</div></Tooltip>))}
                            </div>
                          </CardBody>
                        </Card>
                      </GridItem>
                    )}
                    {selectedMetrics.includes('runs') && (
                      <GridItem span={selectedMetrics.length === 1 ? 12 : 6}>
                        <Card isFlat style={{ height: '100%' }}>
                          <CardTitle style={{ fontSize: '1rem', paddingBottom: '0.5rem' }}><TachometerAltIcon style={{ marginRight: '0.5rem', color: '#8a8d90' }} />Number of Runs</CardTitle>
                          <CardBody style={{ paddingTop: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', height: '180px', paddingBottom: '0.75rem', borderBottom: '1px solid #d2d2d2' }}>
                              {flowStatistics.byFlow.map((stat) => (<Tooltip key={stat.configId} content={`${stat.flowName}: ${stat.runCount} runs`}><div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, maxWidth: '80px' }}><div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.3rem', color: '#151515' }}>{stat.runCount}</div><div style={{ width: '36px', height: `${Math.max((stat.runCount / flowStatistics.maxRuns) * 130, 6)}px`, backgroundColor: '#8a8d90', borderRadius: '4px 4px 0 0', transition: 'height 0.3s ease' }} /></div></Tooltip>))}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '0.5rem' }}>
                              {flowStatistics.byFlow.map((stat, idx) => (<Tooltip key={stat.configId} content={stat.flowName}><div style={{ flex: 1, maxWidth: '80px', textAlign: 'center', fontSize: '0.8rem', color: '#6a6e73' }}>Flow {idx + 1}</div></Tooltip>))}
                            </div>
                          </CardBody>
                        </Card>
                      </GridItem>
                    )}
                  </Grid>
                )}
              </GridItem>
            </Grid>
          )}
        </CollapsibleSection>
      )}

      {/* Preprocessed Datasets Section */}
      <CollapsibleSection
        title="Preprocessed Datasets"
        icon={DatabaseIcon}
        badge={datasets.length > 0 ? datasets.length : null}
        isCollapsed={collapsedSections.datasets}
        onToggle={() => toggleSection('datasets')}
        headerActions={
          datasets.length > 0 && (
            <SearchInput
              placeholder="Search datasets..."
              value={datasetSearchValue}
              onChange={(_, value) => setDatasetSearchValue(value)}
              onClear={() => setDatasetSearchValue('')}
              style={{ width: '300px' }}
            />
          )
        }
      >
        {isLoadingDatasets ? (
          <div style={{ textAlign: 'center', padding: '3rem' }}><Spinner size="xl" /></div>
        ) : datasets.length === 0 ? (
          <EmptyState variant="sm">
            <EmptyStateIcon icon={DatabaseIcon} />
            <Title headingLevel="h4" size="lg">No Preprocessed Datasets</Title>
            <EmptyStateBody style={{ fontSize: '1rem' }}>Create datasets from PDF documents using the PDF preprocessing pipeline in the configuration wizard.</EmptyStateBody>
          </EmptyState>
        ) : filteredDatasets.length === 0 ? (
          <Text style={{ textAlign: 'center', padding: '3rem', color: '#6a6e73', fontSize: '1.1rem' }}>No datasets match your search.</Text>
        ) : (
          <DataList aria-label="Preprocessed datasets" style={{ '--pf-v5-c-data-list__item-row--PaddingTop': '1rem', '--pf-v5-c-data-list__item-row--PaddingBottom': '1rem' }}>
            {filteredDatasets.map((dataset) => (
              <DataListItem key={dataset.job_id} aria-labelledby={`dataset-${dataset.job_id}`}>
                <DataListItemRow>
                  <DataListItemCells
                    dataListCells={[
                      <DataListCell key="name" width={3}>
                        <Text style={{ fontWeight: 600, fontSize: '1.05rem' }}>{dataset.display_name || dataset.name || 'Unnamed Dataset'}</Text>
                        <Text style={{ fontSize: '0.9rem', color: '#6a6e73', marginTop: '0.25rem' }}>{dataset.source_file}</Text>
                      </DataListCell>,
                      <DataListCell key="info" width={2}>
                        <Text style={{ fontSize: '1rem' }}>
                          <Badge isRead style={{ marginRight: '0.75rem', fontSize: '0.9rem', padding: '0.25rem 0.5rem' }}>{dataset.sample_count || 0} samples</Badge>
                          {new Date(dataset.created_at).toLocaleDateString()}
                        </Text>
                      </DataListCell>,
                    ]}
                  />
                  <DataListAction aria-labelledby={`dataset-${dataset.job_id}`} aria-label="Actions" id={`dataset-actions-${dataset.job_id}`}>
                    <Flex style={{ gap: '0.75rem' }}>
                      <FlexItem>
                        <Tooltip content="Download JSONL dataset">
                          <Button variant="secondary" icon={<DownloadIcon />} onClick={() => handleDownloadDataset(dataset.job_id)} style={{ fontSize: '0.95rem' }}>
                            Download
                          </Button>
                        </Tooltip>
                      </FlexItem>
                      <FlexItem>
                        <Tooltip content="Delete dataset">
                          <Button variant="danger" icon={<TrashIcon />} onClick={() => { setDatasetToDelete(dataset); setDeleteModalOpen(true); }} style={{ fontSize: '0.95rem' }}>
                            Delete
                          </Button>
                        </Tooltip>
                      </FlexItem>
                    </Flex>
                  </DataListAction>
                </DataListItemRow>
              </DataListItem>
            ))}
          </DataList>
        )}
      </CollapsibleSection>

      {/* My Custom Flows Section */}
      <CollapsibleSection
        title="My Custom Flows"
        icon={FolderIcon}
        badge={customFlows.length > 0 ? customFlows.length : null}
        isCollapsed={collapsedSections.customFlows}
        onToggle={() => toggleSection('customFlows')}
        headerActions={
          customFlows.length > 0 && (
            <Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '1rem' }}>
              <FlexItem>
                <SearchInput
                  placeholder="Search custom flows..."
                  value={customFlowSearchValue}
                  onChange={(_, value) => setCustomFlowSearchValue(value)}
                  onClear={() => setCustomFlowSearchValue('')}
                  style={{ width: '300px' }}
                />
              </FlexItem>
              <FlexItem>
                <Button 
                  variant="danger" 
                  icon={<TrashIcon />} 
                  onClick={() => setDeleteAllCustomFlowsModalOpen(true)}
                >
                  Delete All Custom Flows
                </Button>
              </FlexItem>
            </Flex>
          )
        }
      >
        {isLoadingCustomFlows ? (
          <div style={{ textAlign: 'center', padding: '3rem' }}><Spinner size="xl" /></div>
        ) : customFlows.length === 0 ? (
          <EmptyState variant="sm">
            <EmptyStateIcon icon={FolderIcon} />
            <Title headingLevel="h4" size="lg">No Custom Flows</Title>
            <EmptyStateBody style={{ fontSize: '1rem' }}>
              Create custom flows using the Visual Flow Builder. Your saved flows will appear here for download.
            </EmptyStateBody>
            <Button variant="primary" onClick={() => onNavigate('flows', { mode: 'create' })} style={{ marginTop: '1rem' }}>
              <PlusCircleIcon style={{ marginRight: '0.5rem' }} />
              Create New Flow
            </Button>
          </EmptyState>
        ) : filteredCustomFlows.length === 0 ? (
          <Text style={{ textAlign: 'center', padding: '3rem', color: '#6a6e73', fontSize: '1.1rem' }}>No custom flows match your search.</Text>
        ) : (
          <DataList aria-label="Custom flows" style={{ '--pf-v5-c-data-list__item-row--PaddingTop': '0.75rem', '--pf-v5-c-data-list__item-row--PaddingBottom': '0.75rem' }}>
            {filteredCustomFlows.map((flow) => (
              <DataListItem key={flow.directory_name} aria-labelledby={`custom-flow-${flow.directory_name}`} isExpanded={expandedCustomFlows[flow.directory_name]}>
                <DataListItemRow>
                  <DataListItemCells
                    dataListCells={[
                      <DataListCell key="name" width={3}>
                        <Flex alignItems={{ default: 'alignItemsCenter' }} style={{ gap: '0.5rem' }}>
                          <FlexItem>
                            <Button 
                              variant="plain" 
                              onClick={() => toggleCustomFlowExpanded(flow.directory_name)}
                              style={{ padding: '0.25rem' }}
                            >
                              {expandedCustomFlows[flow.directory_name] ? <AngleDownIcon /> : <AngleRightIcon />}
                            </Button>
                          </FlexItem>
                          <FlexItem>
                            <div>
                              <Text style={{ fontWeight: 600, fontSize: '1.05rem' }}>{flow.name}</Text>
                              {flow.description && (
                                <Text style={{ fontSize: '0.85rem', color: '#6a6e73', marginTop: '0.15rem' }}>
                                  {flow.description.length > 100 ? flow.description.substring(0, 100) + '...' : flow.description}
                                </Text>
                              )}
                            </div>
                          </FlexItem>
                        </Flex>
                      </DataListCell>,
                      <DataListCell key="info" width={2}>
                        <Flex style={{ gap: '0.5rem' }} alignItems={{ default: 'alignItemsCenter' }}>
                          <Badge isRead style={{ fontSize: '0.85rem' }}>{flow.file_count} files</Badge>
                          {flow.version && <Label color="blue" style={{ fontSize: '0.8rem' }}>v{flow.version}</Label>}
                        </Flex>
                      </DataListCell>,
                    ]}
                  />
                  <DataListAction aria-labelledby={`custom-flow-${flow.directory_name}`} aria-label="Actions" id={`custom-flow-actions-${flow.directory_name}`}>
                    <Flex style={{ gap: '0.5rem' }}>
                      <FlexItem>
                        <Tooltip content="Download all files as ZIP">
                          <Button variant="secondary" icon={<DownloadIcon />} onClick={() => handleDownloadCustomFlowZip(flow.directory_name)} style={{ fontSize: '0.9rem' }}>
                            Download ZIP
                          </Button>
                        </Tooltip>
                      </FlexItem>
                      <FlexItem>
                        <Tooltip content="Delete this custom flow">
                          <Button variant="danger" icon={<TrashIcon />} onClick={() => { setCustomFlowToDelete(flow); setDeleteCustomFlowModalOpen(true); }} style={{ fontSize: '0.9rem' }}>
                            Delete
                          </Button>
                        </Tooltip>
                      </FlexItem>
                    </Flex>
                  </DataListAction>
                </DataListItemRow>
                {expandedCustomFlows[flow.directory_name] && (
                  <div style={{ padding: '0.75rem 1rem 1rem 3rem', backgroundColor: '#f5f5f5', borderTop: '1px solid #d2d2d2' }}>
                    <Text style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem' }}>
                      <FileIcon style={{ marginRight: '0.5rem' }} />
                      YAML Files
                    </Text>
                    <Grid hasGutter style={{ gap: '0.5rem' }}>
                      {flow.files.map((file) => (
                        <GridItem key={file.filename} span={4}>
                          <Card isFlat isCompact style={{ padding: '0.5rem 0.75rem' }}>
                            <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} alignItems={{ default: 'alignItemsCenter' }}>
                              <FlexItem>
                                <Text style={{ fontSize: '0.9rem', fontWeight: 500 }}>{file.filename}</Text>
                                <Text style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                                  {file.size_bytes < 1024 
                                    ? `${file.size_bytes} B` 
                                    : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                </Text>
                              </FlexItem>
                              <FlexItem>
                                <Tooltip content={`Download ${file.filename}`}>
                                  <Button 
                                    variant="plain" 
                                    icon={<DownloadIcon />} 
                                    onClick={() => handleDownloadCustomFlowFile(flow.directory_name, file.filename)}
                                    style={{ padding: '0.25rem' }}
                                  />
                                </Tooltip>
                              </FlexItem>
                            </Flex>
                          </Card>
                        </GridItem>
                      ))}
                    </Grid>
                    {flow.tags && flow.tags.length > 0 && (
                      <div style={{ marginTop: '0.75rem' }}>
                        <Text style={{ fontSize: '0.85rem', color: '#6a6e73', marginBottom: '0.25rem' }}>Tags:</Text>
                        <Flex style={{ gap: '0.25rem' }}>
                          {flow.tags.map((tag) => (
                            <Label key={tag} color="grey" style={{ fontSize: '0.8rem' }}>{tag}</Label>
                          ))}
                        </Flex>
                      </div>
                    )}
                  </div>
                )}
              </DataListItem>
            ))}
          </DataList>
        )}
      </CollapsibleSection>

      {/* Delete Dataset Confirmation Modal */}
      <Modal
        variant={ModalVariant.small}
        title="Delete Dataset"
        isOpen={deleteModalOpen}
        onClose={() => { setDeleteModalOpen(false); setDatasetToDelete(null); }}
        actions={[
          <Button key="delete" variant="danger" onClick={handleDeleteDataset} isLoading={isDeleting} isDisabled={isDeleting}>Delete</Button>,
          <Button key="cancel" variant="link" onClick={() => { setDeleteModalOpen(false); setDatasetToDelete(null); }} isDisabled={isDeleting}>Cancel</Button>,
        ]}
      >
        <Text>Are you sure you want to delete the dataset <strong>{datasetToDelete?.name}</strong>? This action cannot be undone.</Text>
      </Modal>
      
      {/* Delete Custom Flow Confirmation Modal */}
      <Modal
        variant={ModalVariant.small}
        title="Delete Custom Flow"
        isOpen={deleteCustomFlowModalOpen}
        onClose={() => { setDeleteCustomFlowModalOpen(false); setCustomFlowToDelete(null); }}
        actions={[
          <Button key="delete" variant="danger" onClick={handleDeleteCustomFlow} isLoading={isDeletingCustomFlow} isDisabled={isDeletingCustomFlow}>Delete</Button>,
          <Button key="cancel" variant="link" onClick={() => { setDeleteCustomFlowModalOpen(false); setCustomFlowToDelete(null); }} isDisabled={isDeletingCustomFlow}>Cancel</Button>,
        ]}
      >
        <Text>
          Are you sure you want to delete the custom flow <strong>{customFlowToDelete?.name}</strong>?
          This will delete all associated YAML files. This action cannot be undone.
        </Text>
      </Modal>
      
      {/* Delete All Custom Flows Confirmation Modal */}
      <Modal
        variant={ModalVariant.small}
        title="Delete All Custom Flows"
        isOpen={deleteAllCustomFlowsModalOpen}
        onClose={() => setDeleteAllCustomFlowsModalOpen(false)}
        actions={[
          <Button key="delete" variant="danger" onClick={handleDeleteAllCustomFlows} isLoading={isDeletingAllCustomFlows} isDisabled={isDeletingAllCustomFlows}>Delete All</Button>,
          <Button key="cancel" variant="link" onClick={() => setDeleteAllCustomFlowsModalOpen(false)} isDisabled={isDeletingAllCustomFlows}>Cancel</Button>,
        ]}
      >
        <Text>
          Are you sure you want to delete <strong>all {customFlows.length} custom flows</strong>?
          This will permanently delete all custom flow directories and their YAML files. This action cannot be undone.
        </Text>
      </Modal>
    </PageSection>
  );
};

export default Dashboard;
