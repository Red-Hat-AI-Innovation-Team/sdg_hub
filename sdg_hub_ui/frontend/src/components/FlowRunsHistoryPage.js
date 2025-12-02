import React, { useState, useEffect, useRef } from 'react';
import {
  PageSection,
  Title,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
  EmptyStateHeader,
  Spinner,
  Alert,
  AlertVariant,
  Toolbar,
  ToolbarContent,
  ToolbarItem,
  SearchInput,
  Button,
  Label,
  Grid,
  GridItem,
  Card,
  CardTitle,
  CardBody,
  Chip,
  ChipGroup,
  Dropdown,
  DropdownList,
  DropdownItem,
  MenuToggle,
  Tooltip,
} from '@patternfly/react-core';
import {
  Table,
  Thead,
  Tr,
  Th,
  Tbody,
  Td,
} from '@patternfly/react-table';
import {
  HistoryIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  InProgressIcon,
  TrashIcon,
  DownloadIcon,
  StopCircleIcon,
  EllipsisVIcon,
  ExternalLinkAltIcon,
} from '@patternfly/react-icons';
import api, { runsAPI, savedConfigAPI, API_BASE_URL } from '../services/api';

/**
 * Run History page
 * Shows history and status of flow executions
 */
const API_BASE = API_BASE_URL.replace(/\/$/, '');

const FlowRunsHistoryPage = () => {
  const [runs, setRuns] = useState([]);
  const [filteredRuns, setFilteredRuns] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchValue, setSearchValue] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [flowTypeFilter, setFlowTypeFilter] = useState('all');
  
  // Advanced multi-tag search state
  const [searchType, setSearchType] = useState('flow_name'); // 'flow_name', 'model', 'status', 'type'
  const [searchTags, setSearchTags] = useState([]);
  const [autocompleteOptions, setAutocompleteOptions] = useState([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const searchContainerRef = useRef(null);
  
  // Actions dropdown state
  const [openActionMenuId, setOpenActionMenuId] = useState(null);
  
  // Configurations state (to check if config still exists)
  const [existingConfigs, setExistingConfigs] = useState([]);

  /**
   * Click outside to close autocomplete
   */
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setShowAutocomplete(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  

  /**
   * Load existing configurations to check which ones still exist
   */
  const loadExistingConfigurations = async () => {
    try {
      const response = await savedConfigAPI.list();
      setExistingConfigs(response.configurations || []);
    } catch (err) {
      setExistingConfigs([]);
    }
  };

  /**
   * Check if a run's configuration still exists
   */
  const configStillExists = (run) => {
    return existingConfigs.some(config => 
      config.flow_name === run.flow_name || 
      config.id === run.config_id
    );
  };

  /**
   * Load runs from backend
   */
  const loadRuns = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await runsAPI.list();
      const sortedRuns = (response.runs || []).sort((a, b) => 
        new Date(b.start_time) - new Date(a.start_time)
      );
      setRuns(sortedRuns);
      setFilteredRuns(sortedRuns);
    } catch (err) {
      console.error('Error loading runs:', err);
      setError('Failed to load run history: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Load runs and configurations on mount
   */
  useEffect(() => {
    loadRuns();
    loadExistingConfigurations();
    const interval = setInterval(() => {
      loadRuns();
      loadExistingConfigurations(); // Also refresh configs
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  /**
   * Get autocomplete suggestions based on search type and input
   */
  const getAutocompleteSuggestions = (type, input) => {
    if (!input || input.length < 1) return [];
    
    const inputLower = input.toLowerCase();
    const uniqueValues = new Set();
    
    runs.forEach(run => {
      switch (type) {
        case 'flow_name':
          if (run.flow_name?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(run.flow_name);
          }
          break;
        case 'model':
          if (run.model_name?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(run.model_name);
          }
          break;
        case 'status': {
          const status = run.status || 'unknown';
          if (status.toLowerCase().includes(inputLower)) {
            uniqueValues.add(status);
          }
          break;
        }
        case 'type':
          if (run.flow_type?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(run.flow_type);
          }
          break;
        default:
          break;
      }
    });
    
    return Array.from(uniqueValues).slice(0, 10);
  };

  /**
   * Update autocomplete options when search value changes
   */
  useEffect(() => {
    if (searchValue && searchValue.length > 0) {
      const suggestions = getAutocompleteSuggestions(searchType, searchValue);
      setAutocompleteOptions(suggestions);
      setShowAutocomplete(suggestions.length > 0);
    } else {
      setShowAutocomplete(false);
      setAutocompleteOptions([]);
    }
  }, [searchValue, searchType, runs]);

  /**
   * Add a search tag
   */
  const addSearchTag = (type, value) => {
    const exists = searchTags.some(tag => tag.type === type && tag.value === value);
    if (!exists) {
      setSearchTags([...searchTags, { type, value }]);
    }
    setSearchValue('');
    setShowAutocomplete(false);
  };

  /**
   * Remove a search tag
   */
  const removeSearchTag = (index) => {
    setSearchTags(searchTags.filter((_, i) => i !== index));
  };

  /**
   * Apply filters - now using search tags
   */
  useEffect(() => {
    let filtered = runs;

    // Apply search tags (multi-criteria)
    if (searchTags.length > 0) {
      filtered = filtered.filter(run => {
        return searchTags.every(tag => {
          switch (tag.type) {
            case 'flow_name':
              return run.flow_name === tag.value;
            case 'model':
              return run.model_name === tag.value;
            case 'status':
              return run.status === tag.value;
            case 'type':
              return run.flow_type === tag.value;
            default:
              return true;
          }
        });
      });
    }

    // Legacy status filter (keep for backwards compatibility)
    if (statusFilter !== 'all') {
      filtered = filtered.filter(run => run.status === statusFilter);
    }

    // Legacy flow type filter
    if (flowTypeFilter !== 'all') {
      filtered = filtered.filter(run => run.flow_type === flowTypeFilter);
    }

    setFilteredRuns(filtered);
  }, [searchTags, statusFilter, flowTypeFilter, runs]);

  /**
   * Restore configuration from run to Flows page
   */
  const handleRestoreConfiguration = async (run) => {
    try {
      
      // Try to get flow details if it's an existing flow
      let flowId = null;
      let flowPath = null;
      
      if (run.flow_type === 'existing' || run.flow_type === 'default') {
        try {
          // Fetch flow details using the info endpoint
          const encodedFlowName = encodeURIComponent(run.flow_name);
          try {
            const { data: flowInfo } = await api.get(`/api/flows/${encodedFlowName}/info`);
            flowId = flowInfo.id;
            flowPath = flowInfo.path;
          } catch (err) {
            // Flow might have been removed; continue with defaults
          }
        } catch (err) {
          // Warning: Could not fetch flow details:', err);
        }
      }
      
      // Extract just the filename from the dataset path
      let datasetFileName = '';
      if (run.dataset_file) {
        // Handle paths like "uploads/file.jsonl" or just "file.jsonl"
        const pathParts = run.dataset_file.split('/');
        datasetFileName = pathParts[pathParts.length - 1];
      }
      
      // Check if we successfully found the flow
      if (!flowId || !flowPath) {
        alert(`❌ Cannot restore configuration: The flow "${run.flow_name}" was not found in the system.\n\nPossible reasons:\n- The flow was deleted or renamed\n- The flow is a custom flow that no longer exists\n- The flow definition was moved\n\nYou may need to create a new configuration for this flow.`);
        return;
      }
      
      // Create configuration object from run data
      const configToRestore = {
        flow_name: run.flow_name,
        flow_id: flowId,
        flow_path: flowPath,
        flow_type: run.flow_type || 'existing',
        model_configuration: {
          model: run.model_name || 'Not configured',
          api_base: 'http://localhost:8001/v1', // Default to mock server
          api_key: 'EMPTY', // Default value
        },
        dataset_configuration: {
          data_files: run.dataset_file || '',
          uploaded_file: datasetFileName, // Add the filename for display
          num_samples: run.input_samples || null,
          split: 'train',
          shuffle: true,
          seed: 42,
        },
        tags: [],
        status: 'configured', // Mark as configured since we have all the data
      };
      
      
      // Save to backend
      const response = await savedConfigAPI.save(configToRestore);
      
      // Reload configurations to reflect the new one
      await loadExistingConfigurations();
      
      // Show success message
      setError(null);
      alert(`✅ Configuration "${run.flow_name}" has been restored to the Flows page!\n\nFlow Details:\n- ID: ${flowId}\n- Dataset: ${datasetFileName || 'None'}\n\nNote: The dataset file may need to be re-uploaded if no longer available on the server. The model API settings have been set to defaults and may need adjustment.`);
      
    } catch (error) {
      console.error('Error restoring configuration:', error);
      setError('Failed to restore configuration: ' + error.message);
      alert(`❌ Failed to restore configuration: ${error.message}`);
    }
  };

  /**
   * Delete a run
   */
  const handleDelete = async (runId) => {
    if (window.confirm('Are you sure you want to delete this run?')) {
      try {
        await runsAPI.delete(runId);
        await loadRuns();
      } catch (err) {
        setError('Failed to delete run: ' + err.message);
      }
    }
  };

  /**
   * Remove all runs
   */
  const handleRemoveAll = async () => {
    if (!window.confirm(`Are you sure you want to delete all ${runs.length} flow runs? This cannot be undone.`)) {
      return;
    }
    
    try {
      // Delete all runs
      await Promise.all(runs.map(run => runsAPI.delete(run.run_id)));
      setRuns([]);
      setFilteredRuns([]);
    } catch (err) {
      console.error('Error removing all runs:', err);
      setError('Failed to remove all runs: ' + err.message);
    }
  };

  /**
   * Get status icon and color
   */
  const getStatusDisplay = (status) => {
    switch (status) {
      case 'completed':
        return { icon: <CheckCircleIcon />, color: 'green', label: 'Completed' };
      case 'failed':
        return { icon: <ExclamationCircleIcon />, color: 'red', label: 'Failed' };
      case 'running':
        return { icon: <InProgressIcon />, color: 'blue', label: 'Running' };
      case 'cancelled':
      case 'stopped':
        return { icon: <StopCircleIcon />, color: 'orange', label: 'Stopped' };
      default:
        return { icon: null, color: 'grey', label: status };
    }
  };

  /**
   * Format date
   */
  const formatDate = (isoString) => {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleString();
  };

  /**
   * Format duration
   */
  const formatDuration = (seconds) => {
    if (!seconds) return '-';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
  };

  /**
   * Calculate stats
   */
  const stats = {
    total: runs.length,
    completed: runs.filter(r => r.status === 'completed').length,
    failed: runs.filter(r => r.status === 'failed').length,
    stopped: runs.filter(r => r.status === 'cancelled' || r.status === 'stopped').length,
    running: runs.filter(r => r.status === 'running').length,
  };

  /**
   * Render stats cards
   */
  const renderStats = () => (
    <Grid hasGutter style={{ marginBottom: '24px' }}>
      <GridItem sm={12} md={6} lg={2}>
        <Card isCompact>
          <CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HistoryIcon />
              <span>Total</span>
            </div>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h2" size="3xl">{stats.total}</Title>
          </CardBody>
        </Card>
      </GridItem>
      
      <GridItem sm={12} md={6} lg={2}>
        <Card isCompact>
          <CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircleIcon style={{ color: 'var(--pf-v5-global--success-color--100)' }} />
              <span>Completed</span>
            </div>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h2" size="3xl">{stats.completed}</Title>
          </CardBody>
        </Card>
      </GridItem>
      
      <GridItem sm={12} md={6} lg={2}>
        <Card isCompact>
          <CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ExclamationCircleIcon style={{ color: 'var(--pf-v5-global--danger-color--100)' }} />
              <span>Failed</span>
            </div>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h2" size="3xl">{stats.failed}</Title>
          </CardBody>
        </Card>
      </GridItem>
      
      <GridItem sm={12} md={6} lg={2}>
        <Card isCompact>
          <CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <StopCircleIcon style={{ color: 'var(--pf-v5-global--warning-color--100)' }} />
              <span>Stopped</span>
            </div>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h2" size="3xl">{stats.stopped}</Title>
          </CardBody>
        </Card>
      </GridItem>
      
      <GridItem sm={12} md={6} lg={2}>
        <Card isCompact>
          <CardTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <InProgressIcon style={{ color: 'var(--pf-v5-global--info-color--100)' }} />
              <span>Running</span>
            </div>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h2" size="3xl">{stats.running}</Title>
          </CardBody>
        </Card>
      </GridItem>
    </Grid>
  );

  /**
   * Get search placeholder based on type
   */
  const getSearchPlaceholder = () => {
    switch (searchType) {
      case 'flow_name':
        return 'Search by flow name';
      case 'model':
        return 'Search by model';
      case 'status':
        return 'Search by status';
      case 'type':
        return 'Search by type';
      default:
        return 'Search runs';
    }
  };

  /**
   * Render toolbar
   */
  const renderToolbar = () => (
    <Toolbar id="runs-toolbar">
      <ToolbarContent>
        <ToolbarItem>
          <select
            value={searchType}
            onChange={(e) => {
              setSearchType(e.target.value);
              setSearchValue('');
              setShowAutocomplete(false);
            }}
            style={{
              padding: '6px 12px',
              borderRadius: '3px',
              border: '1px solid #d2d2d2',
              marginRight: '8px',
              height: '36px',
            }}
          >
            <option value="flow_name">Flow Name</option>
            <option value="model">Model</option>
            <option value="status">Status</option>
            <option value="type">Type</option>
          </select>
        </ToolbarItem>
        <ToolbarItem variant="search-filter" widths={{ default: '400px' }}>
          <div ref={searchContainerRef} style={{ position: 'relative', width: '100%' }}>
            <SearchInput
              placeholder={getSearchPlaceholder()}
              value={searchValue}
              onChange={(_event, value) => setSearchValue(value)}
              onClear={() => {
                setSearchValue('');
                setShowAutocomplete(false);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && searchValue) {
                  addSearchTag(searchType, searchValue);
                }
              }}
            />
            {/* Autocomplete Dropdown */}
            {showAutocomplete && autocompleteOptions.length > 0 && (
              <div style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                zIndex: 1000,
                backgroundColor: 'white',
                border: '1px solid #d2d2d2',
                borderRadius: '4px',
                marginTop: '4px',
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                maxHeight: '300px',
                overflowY: 'auto'
              }}>
                {autocompleteOptions.map((option, idx) => (
                  <div
                    key={idx}
                    onClick={() => addSearchTag(searchType, option)}
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      borderBottom: idx < autocompleteOptions.length - 1 ? '1px solid #f0f0f0' : 'none',
                      transition: 'background-color 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f9ff'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
                  >
                    <div style={{ fontSize: '0.875rem', fontWeight: '500' }}>
                      {option}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginTop: '2px' }}>
                      Click to add as {searchType === 'flow_name' ? 'flow' : searchType} filter
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ToolbarItem>
        <ToolbarItem align={{ default: 'alignRight' }}>
          <Button variant="secondary" onClick={loadRuns}>
            Refresh
          </Button>
        </ToolbarItem>
        <ToolbarItem>
          <Button 
            variant="danger" 
            icon={<TrashIcon />}
            onClick={handleRemoveAll}
          >
            Remove All
          </Button>
        </ToolbarItem>
      </ToolbarContent>
      
      {/* Search Tags Row - Inside toolbar white background */}
      {searchTags.length > 0 && (
        <ToolbarContent style={{ paddingTop: '0', paddingBottom: '12px' }}>
          <ToolbarItem style={{ width: '100%' }}>
            <ChipGroup categoryName="Active Filters" numChips={10}>
              {searchTags.map((tag, index) => (
                <Chip
                  key={index}
                  onClick={() => removeSearchTag(index)}
                >
                  <strong>{tag.type === 'flow_name' ? 'Flow' : tag.type.charAt(0).toUpperCase() + tag.type.slice(1)}:</strong> {tag.value}
                </Chip>
              ))}
            </ChipGroup>
          </ToolbarItem>
        </ToolbarContent>
      )}
    </Toolbar>
  );

  /**
   * Render empty state
   */
  const renderEmptyState = () => (
    <EmptyState>
      <EmptyStateHeader
        titleText="No flow runs yet"
        icon={<EmptyStateIcon icon={HistoryIcon} />}
        headingLevel="h2"
      />
      <EmptyStateBody>
        Run a flow from the Flows page to see execution history here.
        All completed and in-progress generation runs will appear in this list.
      </EmptyStateBody>
    </EmptyState>
  );

  /**
   * Render loading state
   */
  if (isLoading && runs.length === 0) {
    return (
      <PageSection isCenterAligned>
        <Spinner size="xl" />
      </PageSection>
    );
  }

  return (
    <>
      {/* Page Header */}
      <PageSection variant="light">
        <Title headingLevel="h1" size="2xl">Flow Runs History</Title>
        <p style={{ 
          marginTop: '8px', 
          color: 'var(--pf-v5-global--Color--200)',
          fontSize: '14px'
        }}>
          Track and monitor the execution history of all your flow generations. 
          View progress, review results, and manage completed runs.
        </p>
      </PageSection>

      <PageSection>
        {error && (
          <Alert
            variant={AlertVariant.danger}
            title="Error"
            isInline
            style={{ marginBottom: '16px' }}
          >
            {error}
          </Alert>
        )}

        {/* Stats */}
        {runs.length > 0 && renderStats()}

        {/* Toolbar */}
        {runs.length > 0 && renderToolbar()}

        {/* Table */}
        {filteredRuns.length === 0 && runs.length > 0 ? (
          <EmptyState>
            <EmptyStateHeader
              titleText="No runs match your filters"
              headingLevel="h2"
            />
            <EmptyStateBody>
              Try adjusting your search or filter criteria.
            </EmptyStateBody>
          </EmptyState>
        ) : runs.length === 0 ? (
          renderEmptyState()
        ) : (
          <Table aria-label="Flow runs table" variant="compact">
            <Thead>
              <Tr>
                <Th>Flow Name</Th>
                <Th>Status</Th>
                <Th>Type</Th>
                <Th>Model</Th>
                <Th>Start Time</Th>
                <Th>Duration</Th>
                <Th>Input Samples</Th>
                <Th>Output Samples</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {filteredRuns.map((run) => {
                const statusDisplay = getStatusDisplay(run.status);
                return (
                  <Tr key={run.run_id}>
                    <Td>
                      <Tooltip content={run.flow_name} position="top">
                        <div style={{ 
                          fontWeight: 600,
                          maxWidth: '200px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          cursor: 'default'
                        }}>
                          {run.flow_name}
                        </div>
                      </Tooltip>
                      <div style={{ fontSize: '0.85em', color: '#6a6e73' }}>
                        {run.run_id}
                      </div>
                    </Td>
                    <Td>
                      <Label color={statusDisplay.color} icon={statusDisplay.icon} isCompact>
                        {statusDisplay.label}
                      </Label>
                    </Td>
                    <Td>
                      <Label color={run.flow_type === 'custom' ? 'purple' : 'blue'} isCompact>
                        {run.flow_type}
                      </Label>
                    </Td>
                    <Td>{run.model_name}</Td>
                    <Td>{formatDate(run.start_time)}</Td>
                    <Td>{formatDuration(run.duration_seconds)}</Td>
                    <Td>{run.input_samples || '-'}</Td>
                    <Td>
                      {run.output_samples ? (
                        <span>
                          {run.output_samples}
                          {run.output_columns && (
                            <span style={{ color: '#6a6e73' }}>
                              {' '}({run.output_columns} cols)
                            </span>
                          )}
                        </span>
                      ) : '-'}
                    </Td>
                    <Td>
                      <Dropdown
                        isOpen={openActionMenuId === run.run_id}
                        onSelect={() => setOpenActionMenuId(null)}
                        onOpenChange={(isOpen) => setOpenActionMenuId(isOpen ? run.run_id : null)}
                        toggle={(toggleRef) => (
                          <MenuToggle
                            ref={toggleRef}
                            variant="plain"
                            onClick={() => setOpenActionMenuId(openActionMenuId === run.run_id ? null : run.run_id)}
                            isExpanded={openActionMenuId === run.run_id}
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
                          {/* Download Dataset */}
                          {run.status === 'completed' && run.output_file && (
                            <DropdownItem
                              key="download"
                            icon={<DownloadIcon />}
                              onClick={() => {
                                runsAPI.download(run.run_id);
                                setOpenActionMenuId(null);
                              }}
                              description="Download generated output file"
                            >
                              Download Dataset
                            </DropdownItem>
                        )}
                          
                          {/* Return to Flows Page - only if config was deleted */}
                          {!configStillExists(run) && (
                            <DropdownItem
                              key="return-to-flows"
                              icon={<ExternalLinkAltIcon />}
                              onClick={async () => {
                                setOpenActionMenuId(null);
                                await handleRestoreConfiguration(run);
                              }}
                              description="Restore this configuration"
                            >
                              Return to Flows Page
                            </DropdownItem>
                          )}
                          
                          {/* Delete Run */}
                        {run.status !== 'running' && (
                            <DropdownItem
                              key="delete"
                            icon={<TrashIcon />}
                              onClick={() => {
                                handleDelete(run.run_id);
                                setOpenActionMenuId(null);
                              }}
                              isDanger
                            >
                              Delete Run
                            </DropdownItem>
                        )}
                        </DropdownList>
                      </Dropdown>
                    </Td>
                  </Tr>
                );
              })}
            </Tbody>
          </Table>
        )}
      </PageSection>

    </>
  );
};

export default FlowRunsHistoryPage;
