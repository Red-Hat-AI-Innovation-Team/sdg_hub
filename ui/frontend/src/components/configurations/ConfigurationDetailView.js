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
  ToggleGroup,
  ToggleGroupItem,
  Spinner,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
  DataList,
  DataListItem,
  DataListItemRow,
  DataListItemCells,
  DataListCell,
} from '@patternfly/react-core';
import { ArrowLeftIcon, InfoCircleIcon, TerminalIcon, ChartLineIcon, CheckCircleIcon, InProgressIcon, StopIcon, PlayIcon, RedoIcon, HistoryIcon, CaretDownIcon, ListIcon, DatabaseIcon, DownloadIcon, ClockIcon, CubesIcon, ExclamationCircleIcon, EyeIcon, EyeSlashIcon } from '@patternfly/react-icons';
import AnsiToHtml from 'ansi-to-html';
import LiveMonitoring from '../LiveMonitoring';
import { runsAPI } from '../../services/api';

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
  const [processViewMode, setProcessViewMode] = useState('visual'); // 'visual' or 'raw'
  const [generatedDatasets, setGeneratedDatasets] = useState([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [expandedDatasetPreviews, setExpandedDatasetPreviews] = useState({}); // { runId: { loading, data, error } }
  const [cellPopup, setCellPopup] = useState(null); // { column, content } for the cell content popup
  
  // Load generated datasets for this configuration
  useEffect(() => {
    const loadGeneratedDatasets = async () => {
      if (!configuration?.id) return;
      setIsLoadingDatasets(true);
      try {
        const response = await runsAPI.listByConfig(configuration.id);
        setGeneratedDatasets(response.runs || []);
      } catch (error) {
        console.error('Failed to load generated datasets:', error);
        setGeneratedDatasets([]);
      } finally {
        setIsLoadingDatasets(false);
      }
    };
    loadGeneratedDatasets();
  }, [configuration?.id]);
  
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
   * Strip ANSI escape codes from a string
   */
  const stripAnsi = (str) => {
    if (!str) return '';
    // eslint-disable-next-line no-control-regex
    return str.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[([0-9;]*)m/g, '');
  };

  /**
   * Render Live Monitoring content (visual progress view)
   */
  const renderLiveMonitoringContent = () => {
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
   * Render Terminal content (raw logs view)
   */
  const renderTerminalContent = () => {
    const hasOutput = executionState && executionState.rawOutput;
    
    // Convert ANSI codes to HTML - strip problematic codes
    const convert = new AnsiToHtml({
      fg: '#c9d1d9',
      bg: '#0d1117',
      newline: true,  // Convert newlines to <br>
      escapeXML: true,
    });
    
    if (!hasOutput) {
      return (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--pf-v5-global--Color--200)' }}>
          <p>Raw logs will appear here when you run Dry Run or Generate.</p>
        </div>
      );
    }
    
    // Clean up the raw output - remove problematic escape sequences
    // and ensure proper newline handling
    let cleanOutput = executionState.rawOutput
      // Remove carriage returns that might cause issues
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      // Remove OSC 8 hyperlink sequences (]8;...\ or \x1b]8;...\x1b\\)
      // These are used by rich/loguru for clickable file links
      .replace(/\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)/g, '')
      .replace(/\]8;[^\\]*\\/g, '')
      .replace(/\]8;;\\/g, '')
      // Remove cursor movement sequences that might cause vertical display
      .replace(/\x1b\[\d*[ABCD]/g, '')  // Cursor up/down/forward/back
      .replace(/\x1b\[\d*[HJK]/g, '')   // Cursor position/clear
      .replace(/\x1b\[\??\d*[hl]/g, '') // Mode set/reset
      .replace(/\x1b\[[\d;]*[suABCDEFGHJKSTfm]/g, (match) => {
        // Keep color codes (ending in 'm'), strip others
        return match.endsWith('m') ? match : '';
      });
    
    // Filter out broken rich console lines (lots of whitespace + single character)
    // These are caused by rich's column formatting not rendering properly
    cleanOutput = cleanOutput.split('\n')
      .filter(line => {
        // Remove lines that are just whitespace + 1-2 characters (broken rich formatting)
        const trimmed = line.trim();
        if (trimmed.length <= 2 && line.length > 10 && /^\s+\S{1,2}\s*$/.test(line)) {
          return false;
        }
        return true;
      })
      .join('\n');
    
    return (
      <>
        <div
          id="terminal-output"
          style={{
            display: 'block',
            width: '100%',
            fontSize: '14px',
            fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace',
            minHeight: '400px',
            maxHeight: '600px',
            overflowY: 'auto',
            overflowX: 'auto',
            backgroundColor: '#0d1117',
            color: '#c9d1d9',
            padding: '1rem',
            borderRadius: '6px',
            border: '1px solid #30363d',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: '1.5',
            boxSizing: 'border-box',
          }}
          dangerouslySetInnerHTML={{
            __html: convert.toHtml(cleanOutput)
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
      </>
    );
  };

  /**
   * Render Running Process tab (combined Terminal + Live Monitoring)
   */
  const renderRunningProcess = () => {
    const hasOutput = executionState && executionState.rawOutput;
    
    return (
      <Card>
        <CardBody>
          {/* Header with title and view toggle */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <Title headingLevel="h2" size="xl">
              {executionState?.isRunning ? '⏳ Running Process' : 'Execution Output'}
            </Title>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              {/* View Mode Toggle */}
              <ToggleGroup aria-label="View mode toggle">
                <ToggleGroupItem
                  text="Visual Progress"
                  icon={<ChartLineIcon />}
                  buttonId="visual-view"
                  isSelected={processViewMode === 'visual'}
                  onChange={() => setProcessViewMode('visual')}
                />
                <ToggleGroupItem
                  text="Raw Logs"
                  icon={<TerminalIcon />}
                  buttonId="raw-view"
                  isSelected={processViewMode === 'raw'}
                  onChange={() => setProcessViewMode('raw')}
                />
              </ToggleGroup>
              
              {/* Clear Terminal Button */}
              {hasOutput && onClearTerminal && processViewMode === 'raw' && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => onClearTerminal(configuration.id)}
                  isDisabled={executionState?.isRunning}
                >
                  Clear Logs
                </Button>
              )}
            </div>
          </div>
          
          {/* Content based on selected view */}
          <div style={{ width: '100%' }}>
            {processViewMode === 'visual' ? renderLiveMonitoringContent() : renderTerminalContent()}
          </div>
        </CardBody>
      </Card>
    );
  };

  /**
   * Format duration in a human-readable way
   */
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };

  /**
   * Handle download for a specific run
   */
  const handleDownloadDataset = async (runId) => {
    try {
      await runsAPI.download(runId);
    } catch (error) {
      console.error('Failed to download dataset:', error);
    }
  };

  /**
   * Toggle preview for a specific run — fetch data on first open
   */
  const handleTogglePreview = async (runId) => {
    const current = expandedDatasetPreviews[runId];
    if (current?.data) {
      // Already loaded — just toggle visibility
      setExpandedDatasetPreviews(prev => {
        const copy = { ...prev };
        if (copy[runId]?.visible) {
          copy[runId] = { ...copy[runId], visible: false };
        } else {
          copy[runId] = { ...copy[runId], visible: true };
        }
        return copy;
      });
      return;
    }

    // Fetch preview
    setExpandedDatasetPreviews(prev => ({ ...prev, [runId]: { loading: true, visible: true } }));
    try {
      const data = await runsAPI.preview(runId, 5);
      setExpandedDatasetPreviews(prev => ({ ...prev, [runId]: { loading: false, visible: true, data } }));
    } catch (error) {
      console.error('Failed to load preview:', error);
      setExpandedDatasetPreviews(prev => ({ ...prev, [runId]: { loading: false, visible: true, error: error.message } }));
    }
  };

  /**
   * Render Generated Datasets tab
   */
  const renderGeneratedDatasets = () => {
    const completedRuns = generatedDatasets.filter(run => run.status === 'completed');
    const totalSamples = completedRuns.reduce((sum, run) => sum + (run.output_samples || 0), 0);
    
    return (
      <Card>
        <CardBody>
          <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} alignItems={{ default: 'alignItemsCenter' }} style={{ marginBottom: '1rem' }}>
            <FlexItem>
              <Title headingLevel="h2" size="xl">
                Generated Datasets
              </Title>
            </FlexItem>
            <FlexItem>
              <Flex style={{ gap: '1rem' }}>
                <FlexItem>
                  <Badge isRead style={{ backgroundColor: '#0066cc', color: 'white' }}>
                    {completedRuns.length} Datasets
                  </Badge>
                </FlexItem>
                <FlexItem>
                  <Badge isRead style={{ backgroundColor: '#3e8635', color: 'white' }}>
                    {totalSamples.toLocaleString()} Total Samples
                  </Badge>
                </FlexItem>
              </Flex>
            </FlexItem>
          </Flex>
          
          {isLoadingDatasets ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <Spinner size="lg" />
            </div>
          ) : generatedDatasets.filter(run => run.status === 'completed').length === 0 ? (
            <EmptyState>
              <EmptyStateIcon icon={CubesIcon} />
              <Title headingLevel="h4" size="lg">
                No Datasets Generated Yet
              </Title>
              <EmptyStateBody>
                Run this flow to generate your first dataset. Each successful generation will appear here.
              </EmptyStateBody>
            </EmptyState>
          ) : (
            <DataList aria-label="Generated datasets list">
              {generatedDatasets.filter(run => run.status === 'completed').map((run, index) => (
                <DataListItem key={run.run_id || index} aria-labelledby={`dataset-${index}`}>
                  <DataListItemRow>
                    <DataListItemCells
                      dataListCells={[
                        <DataListCell key="status" width={1}>
                          {run.status === 'completed' ? (
                            <Label color="green" icon={<CheckCircleIcon />}>Completed</Label>
                          ) : run.status === 'failed' ? (
                            <Label color="red" icon={<ExclamationCircleIcon />}>Failed</Label>
                          ) : run.status === 'running' ? (
                            <Label color="blue" icon={<InProgressIcon />}>Running</Label>
                          ) : (
                            <Label color="grey">{run.status}</Label>
                          )}
                        </DataListCell>,
                        <DataListCell key="time" width={2}>
                          <div>
                            <ClockIcon style={{ marginRight: '0.25rem', color: '#6a6e73' }} />
                            <strong>{new Date(run.start_time).toLocaleString()}</strong>
                          </div>
                          {run.duration_seconds && (
                            <div style={{ fontSize: '0.85rem', color: '#6a6e73', marginTop: '0.25rem' }}>
                              Duration: {formatDuration(run.duration_seconds)}
                            </div>
                          )}
                        </DataListCell>,
                        <DataListCell key="samples" width={2}>
                          {run.status === 'completed' && (
                            <div>
                              <DatabaseIcon style={{ marginRight: '0.25rem', color: '#3e8635' }} />
                              <strong>{run.output_samples?.toLocaleString() || 0}</strong> samples
                              {run.output_columns && (
                                <span style={{ color: '#6a6e73' }}> • {run.output_columns} columns</span>
                              )}
                            </div>
                          )}
                          {run.status === 'failed' && run.error_message && (
                            <div style={{ color: '#c9190b', fontSize: '0.85rem' }}>
                              {run.error_message.substring(0, 50)}...
                            </div>
                          )}
                        </DataListCell>,
                        <DataListCell key="actions" width={2} alignRight>
                          {run.status === 'completed' && run.output_file && (
                            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                              <Button 
                                variant="secondary" 
                                isSmall
                                icon={expandedDatasetPreviews[run.run_id]?.visible ? <EyeSlashIcon /> : <EyeIcon />}
                                onClick={() => handleTogglePreview(run.run_id)}
                              >
                                {expandedDatasetPreviews[run.run_id]?.visible ? 'Hide' : 'View'}
                              </Button>
                              <Button 
                                variant="secondary" 
                                isSmall
                                icon={<DownloadIcon />}
                                onClick={() => handleDownloadDataset(run.run_id)}
                              >
                                Download
                              </Button>
                            </div>
                          )}
                        </DataListCell>,
                      ]}
                    />
                  </DataListItemRow>
                  {/* Expandable preview section */}
                  {expandedDatasetPreviews[run.run_id]?.visible && (
                    <div style={{ 
                      padding: '1rem 1.5rem', 
                      borderTop: '1px solid #d2d2d2',
                      backgroundColor: '#fafafa',
                    }}>
                      {expandedDatasetPreviews[run.run_id]?.loading ? (
                        <div style={{ textAlign: 'center', padding: '1rem' }}>
                          <Spinner size="md" /> <span style={{ marginLeft: '0.5rem', color: '#6a6e73' }}>Loading preview...</span>
                        </div>
                      ) : expandedDatasetPreviews[run.run_id]?.error ? (
                        <div style={{ color: '#c9190b', fontSize: '0.9rem' }}>
                          Failed to load preview: {expandedDatasetPreviews[run.run_id].error}
                        </div>
                      ) : expandedDatasetPreviews[run.run_id]?.data ? (
                        (() => {
                          const previewState = expandedDatasetPreviews[run.run_id];
                          const preview = previewState.data;
                          const selectedCols = previewState.selectedColumns || (preview.columns.length > 0 ? [preview.columns[0]] : []);

                          const toggleColumn = (col) => {
                            setExpandedDatasetPreviews(prev => {
                              const current = prev[run.run_id];
                              const currentSelected = current.selectedColumns || (preview.columns.length > 0 ? [preview.columns[0]] : []);
                              const isSelected = currentSelected.includes(col);
                              let newSelected;
                              if (isSelected) {
                                newSelected = currentSelected.filter(c => c !== col);
                              } else {
                                // Maintain original column order
                                newSelected = preview.columns.filter(c => currentSelected.includes(c) || c === col);
                              }
                              return { ...prev, [run.run_id]: { ...current, selectedColumns: newSelected } };
                            });
                          };

                          const selectAll = () => {
                            setExpandedDatasetPreviews(prev => ({
                              ...prev,
                              [run.run_id]: { ...prev[run.run_id], selectedColumns: [...preview.columns] }
                            }));
                          };

                          const selectNone = () => {
                            setExpandedDatasetPreviews(prev => ({
                              ...prev,
                              [run.run_id]: { ...prev[run.run_id], selectedColumns: [] }
                            }));
                          };

                          return (
                            <div>
                              {/* Column selector */}
                              <div style={{ marginBottom: '0.75rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem' }}>
                                  <strong style={{ fontSize: '0.85rem', color: '#333' }}>
                                    Columns ({preview.columns.length}):
                                  </strong>
                                  <span style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                                    {selectedCols.length} selected
                                  </span>
                                  <Button variant="link" isSmall isInline style={{ fontSize: '0.75rem', padding: 0 }} onClick={selectAll}>
                                    Select All
                                  </Button>
                                  <Button variant="link" isSmall isInline style={{ fontSize: '0.75rem', padding: 0 }} onClick={selectNone}>
                                    Clear
                                  </Button>
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                                  {preview.columns.map(col => {
                                    const isActive = selectedCols.includes(col);
                                    return (
                                      <span
                                        key={col}
                                        onClick={() => toggleColumn(col)}
                                        style={{
                                          backgroundColor: isActive ? '#0066cc' : '#f0f0f0',
                                          color: isActive ? '#fff' : '#6a6e73',
                                          border: isActive ? '1px solid #004080' : '1px solid #d2d2d2',
                                          borderRadius: '4px',
                                          padding: '0.15rem 0.5rem',
                                          fontSize: '0.8rem',
                                          fontFamily: "'Red Hat Mono', monospace",
                                          fontWeight: 500,
                                          cursor: 'pointer',
                                          transition: 'all 0.15s',
                                          userSelect: 'none',
                                        }}
                                      >
                                        {col}
                                      </span>
                                    );
                                  })}
                                </div>
                              </div>

                              {/* Sample rows table */}
                              {selectedCols.length > 0 && (
                                <>
                                  <div style={{ fontSize: '0.85rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                                    Showing {preview.preview_rows} of {preview.total_rows.toLocaleString()} rows
                                  </div>
                                  <div style={{ 
                                    maxHeight: '500px', 
                                    overflowY: 'auto', 
                                    overflowX: 'auto',
                                    border: '1px solid #d2d2d2', 
                                    borderRadius: '6px',
                                    backgroundColor: '#fff',
                                  }}>
                                    <table style={{ 
                                      width: '100%', 
                                      borderCollapse: 'collapse', 
                                      fontSize: '0.8rem',
                                      fontFamily: "'Red Hat Mono', monospace",
                                    }}>
                                      <thead>
                                        <tr style={{ backgroundColor: '#f0f0f0', position: 'sticky', top: 0, zIndex: 1 }}>
                                          <th style={{ 
                                            padding: '0.5rem 0.75rem', 
                                            textAlign: 'left', 
                                            borderBottom: '2px solid #d2d2d2',
                                            fontWeight: 600,
                                            color: '#333',
                                            fontSize: '0.75rem',
                                            whiteSpace: 'nowrap',
                                          }}>
                                            #
                                          </th>
                                          {selectedCols.map(col => (
                                            <th
                                              key={col}
                                              style={{ 
                                                padding: '0.5rem 0.75rem', 
                                                textAlign: 'left', 
                                                borderBottom: '2px solid #d2d2d2',
                                                fontWeight: 600,
                                                color: '#333',
                                                fontSize: '0.75rem',
                                                whiteSpace: 'nowrap',
                                              }}
                                            >
                                              {col}
                                            </th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {preview.rows.map((row, rowIdx) => (
                                          <tr key={rowIdx} style={{ borderBottom: '1px solid #ececec' }}>
                                            <td style={{ 
                                              padding: '0.4rem 0.75rem', 
                                              color: '#6a6e73',
                                              whiteSpace: 'nowrap',
                                              verticalAlign: 'top',
                                            }}>
                                              {rowIdx + 1}
                                            </td>
                                            {selectedCols.map(col => {
                                              const val = row[col];
                                              const display = val === null || val === undefined 
                                                ? '' 
                                                : typeof val === 'object' 
                                                  ? JSON.stringify(val) 
                                                  : String(val);
                                              const truncated = display.length > 80 
                                                ? display.slice(0, 80) + '...' 
                                                : display;
                                              return (
                                                <td
                                                  key={col}
                                                  onClick={() => display && setCellPopup({ column: col, content: display })}
                                                  style={{ 
                                                    padding: '0.4rem 0.75rem',
                                                    color: '#151515',
                                                    whiteSpace: 'nowrap',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                    maxWidth: '250px',
                                                    cursor: display ? 'pointer' : 'default',
                                                  }}
                                                  onMouseEnter={(e) => { if (display) e.currentTarget.style.fontWeight = '600'; }}
                                                  onMouseLeave={(e) => { e.currentTarget.style.fontWeight = 'normal'; }}
                                                >
                                                  {truncated}
                                                </td>
                                              );
                                            })}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                </>
                              )}
                              {selectedCols.length === 0 && (
                                <div style={{ padding: '1rem', textAlign: 'center', color: '#6a6e73', fontSize: '0.9rem' }}>
                                  Select one or more columns above to see the data
                                </div>
                              )}
                            </div>
                          );
                        })()
                      ) : null}
                    </div>
                  )}
                </DataListItem>
              ))}
            </DataList>
          )}
        </CardBody>
      </Card>
    );
  };

  /**
   * Automatically switch to Running Process tab when execution starts or when opening a running config
   */
  useEffect(() => {
    if (executionState && executionState.isRunning) {
      setActiveTabKey('process');
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
      {/* Cell content popup overlay */}
      {cellPopup && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={() => setCellPopup(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: '#fff',
              borderRadius: '8px',
              boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
              width: '90%',
              maxWidth: '700px',
              maxHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Popup header */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.75rem 1rem',
              borderBottom: '1px solid #d2d2d2',
              backgroundColor: '#f0f0f0',
              flexShrink: 0,
            }}>
              <span style={{
                fontWeight: 600,
                fontSize: '0.9rem',
                color: '#0066cc',
                fontFamily: "'Red Hat Mono', monospace",
              }}>
                {cellPopup.column}
              </span>
              <button
                onClick={() => setCellPopup(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.25rem',
                  cursor: 'pointer',
                  color: '#6a6e73',
                  padding: '0 0.25rem',
                  lineHeight: 1,
                }}
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            {/* Popup body */}
            <div style={{
              padding: '1rem',
              overflowY: 'auto',
              fontSize: '0.85rem',
              fontFamily: "'Red Hat Mono', monospace",
              lineHeight: 1.7,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: '#151515',
            }}>
              {cellPopup.content}
            </div>
          </div>
        </div>
      )}
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
              eventKey="process"
              title={<TabTitleText><ListIcon /> Running Process</TabTitleText>}
              aria-label="Running process tab"
            >
              {renderRunningProcess()}
            </Tab>
            
            <Tab
              eventKey="datasets"
              title={
                <TabTitleText>
                  <DatabaseIcon /> Generated Datasets
                  {generatedDatasets.filter(r => r.status === 'completed').length > 0 && (
                    <Badge isRead style={{ marginLeft: '0.5rem' }}>
                      {generatedDatasets.filter(r => r.status === 'completed').length}
                    </Badge>
                  )}
                </TabTitleText>
              }
              aria-label="Generated datasets tab"
            >
              {renderGeneratedDatasets()}
            </Tab>
          </Tabs>
        </FlexItem>
      </Flex>
    </PageSection>
  );
};

export default ConfigurationDetailView;

