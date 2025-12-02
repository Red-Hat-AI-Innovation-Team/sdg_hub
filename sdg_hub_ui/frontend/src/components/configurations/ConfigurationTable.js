import React, { useState } from 'react';
import {
  Checkbox,
  Label,
  LabelGroup,
  Dropdown,
  DropdownList,
  DropdownItem,
  MenuToggle,
  Button,
  Tooltip,
} from '@patternfly/react-core';
import { Table, Thead, Tbody, Tr, Th, Td } from '@patternfly/react-table';
import { 
  EllipsisVIcon, 
  PlayIcon, 
  VialIcon, 
  EditIcon, 
  TrashIcon, 
  StopIcon, 
  InProgressIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  MinusIcon,
  DownloadIcon,
  CopyIcon,
} from '@patternfly/react-icons';
import { runsAPI } from '../../services/api';

/**
 * Configuration Table Component
 * Displays configurations in a table format with columns
 */
const ConfigurationTable = ({
  configurations,
  selectedConfigs,
  onToggleSelection,
  onToggleSelectAll,
  onDryRun,
  onGenerate,
  onEdit,
  onDelete,
  onStop,
  onClone,
  isRunning,
  onFlowNameClick,
  executionStates = {},
}) => {
  const [openMenus, setOpenMenus] = useState({});

  const toggleMenu = (configId) => {
    setOpenMenus(prev => ({
      ...prev,
      [configId]: !prev[configId]
    }));
  };

  const getModelName = (config) => {
    const modelConfig = config.model_configuration || config.model_config || {};
    return modelConfig.model || 'Not configured';
  };

  /**
   * Strip ANSI escape codes from a string
   */
  const stripAnsi = (str) => {
    if (!str) return '';
    // Use RegExp constructor to avoid raw control characters in regex literals
    const ansiPattern = new RegExp('\\x1b\\[[0-9;]*m', 'g');
    const bracketPattern = new RegExp('\\[([0-9;]*)m', 'g');
    return str.replace(ansiPattern, '').replace(bracketPattern, '');
  };

  /**
   * Get block progress from execution state rawOutput
   */
  const getBlockProgress = (configId) => {
    const executionState = executionStates[configId];
    if (!executionState) {
      return null;
    }

    // Check rawOutput (string) which contains the streaming logs - strip ANSI codes
    const rawOutput = stripAnsi(executionState.rawOutput || '');
    
    // If no output yet, return null
    if (!rawOutput) {
      return null;
    }
    
    let totalBlocks = 0;
    let currentBlock = 0;
    let currentBlockName = '';

    // Parse rawOutput to find block progress
    // Match "Executing block X/Y: BlockName" pattern (with optional whitespace and underscore support)
    const blockMatches = [...rawOutput.matchAll(/Executing block (\d+)\/(\d+):\s*([\w_]+)/g)];
    for (const match of blockMatches) {
      currentBlock = parseInt(match[1]);
      totalBlocks = parseInt(match[2]);
      currentBlockName = match[3];
    }
    
    // Calculate completed blocks based on current block number
    // If executing block 5, then blocks 1-4 are completed (currentBlock - 1)
    let completedBlocks = currentBlock > 0 ? currentBlock - 1 : 0;

    // Also check for completion status
    if (executionState.status === 'completed' && totalBlocks > 0) {
      completedBlocks = totalBlocks;
      currentBlockName = 'Complete';
    }

    // If we found block progress info
    if (totalBlocks > 0) {
      // Cap completed blocks at total (safety check)
      completedBlocks = Math.min(completedBlocks, totalBlocks);
      const percentage = Math.round((completedBlocks / totalBlocks) * 100);
      return { 
        completed: completedBlocks, 
        total: totalBlocks,
        current: currentBlock,
        currentBlockName: currentBlockName,
        percentage: percentage,
        isComplete: completedBlocks === totalBlocks
      };
    }
    
    return null;
  };

  /**
   * Progress Bar Component
   */
  const ProgressBar = ({ progress }) => {
    if (!progress) {
      return <span style={{ color: '#6a6e73', fontSize: '13px' }}>-</span>;
    }

    const { completed, total, currentBlockName, percentage, isComplete } = progress;
    
    const tooltipContent = (
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontWeight: 600, marginBottom: '4px' }}>
          {isComplete ? '✅ Complete' : `🔄 ${currentBlockName || 'Processing...'}`}
        </div>
        <div style={{ fontSize: '12px', opacity: 0.9 }}>
          {completed}/{total} blocks ({percentage}%)
        </div>
      </div>
    );

    return (
      <Tooltip content={tooltipContent} position="top">
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '6px',
          cursor: 'pointer',
          width: '100px'
        }}>
          {/* Progress bar container */}
          <div style={{
            width: '50px',
            height: '6px',
            backgroundColor: '#e0e0e0',
            borderRadius: '3px',
            overflow: 'hidden',
            flexShrink: 0
          }}>
            {/* Progress fill */}
            <div style={{
              width: `${percentage}%`,
              height: '100%',
              backgroundColor: isComplete ? '#3e8635' : '#0066cc',
              borderRadius: '3px',
              transition: 'width 0.3s ease-in-out'
            }} />
          </div>
          {/* Progress text */}
          <span style={{ 
            fontSize: '11px', 
            color: isComplete ? '#3e8635' : '#6a6e73',
            fontWeight: 500,
            whiteSpace: 'nowrap'
          }}>
            {completed}/{total}
          </span>
        </div>
      </Tooltip>
    );
  };

  const getDatasetName = (config) => {
    const datasetConfig = config.dataset_configuration || config.dataset_config || {};
    if (datasetConfig?.data_files) {
      const path = datasetConfig.data_files;
      const parts = path.split('/');
      return parts[parts.length - 1];
    }
    return 'Not specified';
  };

  const allSelected = configurations.length > 0 && selectedConfigs.length === configurations.length;
  const someSelected = selectedConfigs.length > 0 && selectedConfigs.length < configurations.length;

  /**
   * Get status display for configuration
   */
  const getStatusDisplay = (config) => {
    // Check for draft status first
    if (config.status === 'draft') {
      return {
        icon: <EditIcon />,
        color: 'purple',
        label: 'Draft'
      };
    }
    
    // Check for not_configured status (flow created but not configured)
    if (config.status === 'not_configured') {
      return {
        icon: <MinusIcon />,
        color: 'orange',
        label: 'Not Configured'
      };
    }
    
    const isConfigRunning = isRunning(config.id);
    const executionState = executionStates[config.id];
    
    // Check current execution state first (live state)
    if (executionState) {
      if (isConfigRunning) {
        // Currently running
        return {
          icon: <InProgressIcon />,
          color: 'blue',
          label: 'Running'
        };
      } else if (executionState.status === 'completed') {
        // Just completed (still in memory)
        return {
          icon: <CheckCircleIcon />,
          color: 'green',
          label: 'Completed'
        };
      } else if (executionState.status === 'failed') {
        // Just failed (still in memory)
        return {
          icon: <ExclamationCircleIcon />,
          color: 'red',
          label: 'Failed'
        };
      } else if (executionState.status === 'cancelled') {
        // Just cancelled (still in memory)
        return {
          icon: <ExclamationCircleIcon />,
          color: 'orange',
          label: 'Stopped'
        };
      }
    }
    
    // If no current execution state, just show as configured and ready
    // We don't look at run history here - that's for the Run History page
    return {
      icon: <CheckCircleIcon />,
      color: 'grey',
      label: 'Ready'
    };
  };

  return (
    <div style={{ marginTop: '16px' }}>
      <Table 
        aria-label="Configurations table" 
        variant="compact" 
        borders={true} 
      >
      <Thead>
        <Tr>
          <Th width={10}>
            <Checkbox
              id="select-all"
              isChecked={allSelected}
              onChange={onToggleSelectAll}
              aria-label="Select all configurations"
              {...(someSelected && { isIndeterminate: true })}
            />
          </Th>
          <Th width={20}>Flow Name</Th>
          <Th width={10}>Status</Th>
          <Th width={10}>Progress</Th>
          <Th width={18}>Model</Th>
          <Th width={10}>Dataset</Th>
          <Th width={12}>Tags</Th>
          <Th width={10} style={{ textAlign: 'center' }}>Actions</Th>
        </Tr>
      </Thead>
      <Tbody>
        {configurations.map((config) => {
          const isConfigRunning = isRunning(config.id);
          const statusDisplay = getStatusDisplay(config);
          const visibleTags = config.tags?.slice(0, 2) || [];
          const hiddenTags = config.tags?.slice(2) || [];
          
          // Check if this config has downloadable output in its current execution state
          const executionState = executionStates[config.id];
          const hasDownloadableOutput = executionState?.result?.output_file || 
                                       (executionState?.status === 'completed' && executionState?.outputFile);
          
          return (
            <Tr key={config.id}>
              <Td>
                <Checkbox
                  id={`select-${config.id}`}
                  isChecked={selectedConfigs.includes(config.id)}
                  onChange={() => onToggleSelection(config.id)}
                  aria-label={`Select ${config.flow_name}`}
                />
              </Td>
              <Td dataLabel="Flow Name">
                <Tooltip content={config.flow_name || config.name} position="top">
                  <Button
                    variant="link"
                    isInline
                    onClick={() => onFlowNameClick(config)}
                    style={{ 
                      fontSize: '14px', 
                      fontWeight: 600, 
                      padding: 0, 
                      textAlign: 'left',
                      maxWidth: '250px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      display: 'block'
                    }}
                  >
                    {config.flow_name || config.name}
                  </Button>
                </Tooltip>
              </Td>
              <Td dataLabel="Status">
                <Label color={statusDisplay.color} icon={statusDisplay.icon} isCompact>
                  {statusDisplay.label}
                </Label>
              </Td>
              <Td dataLabel="Progress">
                <ProgressBar progress={getBlockProgress(config.id)} />
              </Td>
              <Td dataLabel="Model">
                <code style={{ fontSize: '12px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'block' }}>
                  {getModelName(config)}
                </code>
              </Td>
              <Td dataLabel="Dataset">
                <span style={{ fontSize: '13px' }}>{getDatasetName(config)}</span>
              </Td>
              <Td dataLabel="Tags">
                {config.tags && config.tags.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'nowrap' }}>
                    {visibleTags.map((tag, idx) => (
                      <Label key={idx} color="blue" isCompact>
                        {tag}
                      </Label>
                    ))}
                    {hiddenTags.length > 0 && (
                      <Tooltip
                        content={
                          <div>
                            {hiddenTags.map((tag, idx) => (
                              <div key={idx} style={{ padding: '2px 0' }}>{tag}</div>
                            ))}
                          </div>
                        }
                      >
                        <Label color="grey" isCompact style={{ cursor: 'pointer' }}>
                          +{hiddenTags.length}
                        </Label>
                      </Tooltip>
                    )}
                  </div>
                )}
              </Td>
              <Td style={{ textAlign: 'center' }}>
                <Dropdown
                  isOpen={openMenus[config.id] || false}
                  onSelect={() => toggleMenu(config.id)}
                  onOpenChange={(isOpen) => setOpenMenus(prev => ({ ...prev, [config.id]: isOpen }))}
                  toggle={(toggleRef) => (
                    <MenuToggle
                      ref={toggleRef}
                      aria-label="Actions"
                      variant="plain"
                      onClick={() => toggleMenu(config.id)}
                      isExpanded={openMenus[config.id]}
                    >
                      <EllipsisVIcon />
                    </MenuToggle>
                  )}
                  shouldFocusToggleOnSelect
                  popperProps={{
                    placement: 'bottom-end'
                  }}
                >
                  <DropdownList>
                    <DropdownItem
                      key="dry-run"
                      icon={<VialIcon />}
                      onClick={() => {
                        onDryRun(config);
                        toggleMenu(config.id);
                      }}
                      isDisabled={isConfigRunning || config.status === 'draft' || config.status === 'not_configured'}
                      description={config.status === 'draft' ? 'Complete flow configuration first' : config.status === 'not_configured' ? 'Configure model and dataset first' : undefined}
                    >
                      Dry Run
                    </DropdownItem>
                    <DropdownItem
                      key="generate"
                      icon={<PlayIcon />}
                      onClick={() => {
                        onGenerate(config);
                        toggleMenu(config.id);
                      }}
                      isDisabled={isConfigRunning || config.status === 'draft' || config.status === 'not_configured'}
                      description={config.status === 'draft' ? 'Complete flow configuration first' : config.status === 'not_configured' ? 'Configure model and dataset first' : undefined}
                    >
                      Run
                    </DropdownItem>
                    {hasDownloadableOutput && executionState?.runId && (
                      <DropdownItem
                        key="download"
                        icon={<DownloadIcon />}
                        onClick={async () => {
                          try {
                            await runsAPI.download(executionState.runId);
                          } catch (error) {
                            const msg = (error && error.message) || String(error) || 'Unknown error';
                            console.error('Failed to download dataset:', msg);
                            alert('Failed to download dataset: ' + msg);
                          } finally {
                            toggleMenu(config.id);
                          }
                        }}
                        description="Download generated output file"
                      >
                        Download Dataset
                      </DropdownItem>
                    )}
                    {isConfigRunning && (
                    <DropdownItem
                      key="stop"
                      icon={<StopIcon />}
                      onClick={() => {
                        onStop(config);
                        toggleMenu(config.id);
                      }}
                    >
                      Stop
                    </DropdownItem>
                    )}
                    <DropdownItem
                      key="edit"
                      icon={<EditIcon />}
                      onClick={() => {
                        onEdit(config);
                        toggleMenu(config.id);
                      }}
                      isDisabled={isConfigRunning}
                    >
                      Edit
                    </DropdownItem>
                    <DropdownItem
                      key="clone"
                      icon={<CopyIcon />}
                      onClick={() => {
                        if (onClone) {
                          onClone(config);
                        }
                        toggleMenu(config.id);
                      }}
                      isDisabled={config.status === 'draft'}
                      description="Create a copy of this configuration"
                    >
                      Clone and Modify
                    </DropdownItem>
                    <DropdownItem
                      key="delete"
                      icon={<TrashIcon />}
                      onClick={() => {
                        onDelete(config.id);
                        toggleMenu(config.id);
                      }}
                    >
                      Delete
                    </DropdownItem>
                  </DropdownList>
                </Dropdown>
              </Td>
            </Tr>
          );
        })}
      </Tbody>
    </Table>
    </div>
  );
};

export default ConfigurationTable;

