import React, { useState, useEffect, useRef } from 'react';
import {
  PageSection,
  Title,
  Text,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
  EmptyStateHeader,
  EmptyStateFooter,
  EmptyStateActions,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  Dropdown,
  DropdownList,
  DropdownItem,
  MenuToggle,
  Modal,
  ModalVariant,
  SearchInput,
  Toolbar,
  ToolbarContent,
  ToolbarItem,
  Grid,
  GridItem,
  Card,
  CardTitle,
  CardBody,
  Form,
  FormGroup,
  Select,
  SelectList,
  SelectOption,
  Flex,
  FlexItem,
  Badge,
  Chip,
  ChipGroup,
  Label,
  Menu,
  MenuContent,
  MenuList,
  MenuItem,
  Tooltip,
} from '@patternfly/react-core';
import { 
  CubesIcon, 
  PlusCircleIcon, 
  CheckCircleIcon, 
  ExclamationCircleIcon, 
  InProgressIcon,
  PlayIcon,
  StopIcon,
  TrashIcon,
  TimesIcon,
} from '@patternfly/react-icons';
import { savedConfigAPI, runsAPI, executionAPI } from '../services/api';
import ConfigurationList from './configurations/ConfigurationList';

/**
 * Flows page for managing flow configurations.
 * Displays empty state initially, then shows list of configurations.
 */
const DataGenerationFlowsPage = ({ executionStates, onUpdateExecutionState, getExecutionState, onNavigate, onEditConfiguration, autoRunConfig, onAutoRunComplete }) => {
  const [configurations, setConfigurations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchValue, setSearchValue] = useState('');
  const [searchType, setSearchType] = useState('flow_name'); // 'flow_name', 'model', 'dataset', 'tags'
  const [isConfigureMenuOpen, setIsConfigureMenuOpen] = useState(false);
  const [isActionsMenuOpen, setIsActionsMenuOpen] = useState(false);
  const [selectedConfigs, setSelectedConfigs] = useState([]);
  const [batchGenerateTrigger, setBatchGenerateTrigger] = useState(0);
  
  // Advanced multi-tag search state
  const [searchTags, setSearchTags] = useState([]); // Array of {type: 'flow_name'|'model'|'dataset'|'tags', value: 'xyz'}
  const [autocompleteOptions, setAutocompleteOptions] = useState([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const searchContainerRef = useRef(null);
  
  // Refs for callbacks to avoid stale closures in useEffect
  const onAutoRunCompleteRef = useRef(onAutoRunComplete);
  const loadConfigurationsRef = useRef(null);

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

  // Use execution states from App.js (passed as props)

  /**
   * Load draft flows from localStorage (multi-draft system)
   * Only load drafts that haven't been saved as flows yet
   */
  const loadDraftFlows = () => {
    try {
      const draftsJson = localStorage.getItem('wizard_drafts');
      if (draftsJson) {
        const drafts = JSON.parse(draftsJson);
        if (Array.isArray(drafts)) {
          // Filter out drafts that only have blocks but no metadata
          // These are incomplete and should be shown as drafts
          const validDrafts = drafts.filter(draft => {
            // Only include drafts that have actual content
            return draft.blocks?.length > 0 || draft.metadata?.name;
          });
          
          // Convert drafts to configuration format
          return validDrafts.map(draft => ({
            id: draft.id,
            name: draft.name || draft.metadata?.name || 'Unnamed Draft',
            flow_name: draft.name || draft.metadata?.name || 'Unnamed Draft',
            status: 'draft',
            isDraft: true,
            draftData: draft,
            created_at: draft.lastModified || draft.created_at || new Date().toISOString(),
            model_configuration: { model: 'Not configured' },
            dataset_configuration: { data_files: 'Not configured' },
          }));
        }
      }
    } catch (error) {
      console.error('Failed to parse drafts:', error);
    }
    return [];
  };
  
  /**
   * Determine configuration status based on actual data
   */
  const determineConfigStatus = (config) => {
    // If status is explicitly set to draft or not_configured, respect it
    if (config.status === 'draft' || config.status === 'not_configured') {
      return config.status;
    }
    
    // If status is explicitly set to something else, validate it
    if (config.status) {
      // Double-check if the status is accurate
      const modelConfig = config.model_configuration || config.model_config || {};
      const datasetConfig = config.dataset_configuration || config.dataset_config || {};
      
      const hasModel = modelConfig.model && 
                       modelConfig.model !== 'Not configured' && 
                       modelConfig.model !== 'Not specified';
      const hasDataset = datasetConfig.data_files && 
                         datasetConfig.data_files !== 'Not configured' && 
                         datasetConfig.data_files !== 'Not specified';
      
      // Override configured status if actually not configured
      if (config.status === 'configured' && (!hasModel || !hasDataset)) {
        return 'not_configured';
      }
      
      return config.status;
    }
    
    // Check if model is configured
    const modelConfig = config.model_configuration || config.model_config || {};
    const hasModel = modelConfig.model && 
                     modelConfig.model !== 'Not configured' && 
                     modelConfig.model !== 'Not specified';
    
    // Check if dataset is configured
    const datasetConfig = config.dataset_configuration || config.dataset_config || {};
    const hasDataset = datasetConfig.data_files && 
                       datasetConfig.data_files !== 'Not configured' && 
                       datasetConfig.data_files !== 'Not specified';
    
    // If both are missing or not specified, it's not configured
    if (!hasModel || !hasDataset) {
      return 'not_configured';
    }
    
    // Otherwise it's properly configured
    return 'configured';
  };
  
  // Keep onAutoRunComplete ref updated
  useEffect(() => {
    onAutoRunCompleteRef.current = onAutoRunComplete;
  }, [onAutoRunComplete]);

  /**
   * Load configurations from backend
   */
  const loadConfigurations = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await savedConfigAPI.list();
      const savedConfigs = response.configurations || [];
      
      // Add status to saved configurations based on actual data
      const configsWithStatus = savedConfigs.map(config => ({
        ...config,
        status: determineConfigStatus(config),
      }));
      
      // Load draft flows
      const draftFlows = loadDraftFlows();
      
      // Filter out drafts that have the same name as saved configurations
      // (Backend configurations take priority over localStorage drafts)
      const savedFlowNames = new Set(configsWithStatus.map(c => c.flow_name || c.name));
      const uniqueDrafts = draftFlows.filter(draft => 
        !savedFlowNames.has(draft.flow_name) && !savedFlowNames.has(draft.name)
      );
      
      
      // Combine saved configs and unique drafts only
      setConfigurations([...uniqueDrafts, ...configsWithStatus]);
    } catch (err) {
      console.error('Error loading configurations:', err);
      setError('Failed to load configurations: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Keep loadConfigurations ref updated
  loadConfigurationsRef.current = loadConfigurations;

  /**
   * Load configurations on mount
   */
  useEffect(() => {
    loadConfigurations();
  }, []);

  // State for auto-expanding a config (used after Save and Run)
  const [autoExpandConfigId, setAutoExpandConfigId] = useState(null);
  // State for pending auto-run (when config isn't loaded yet)
  const [pendingAutoRun, setPendingAutoRun] = useState(null);

  /**
   * Handle auto-run after Save and Run from wizard
   */
  useEffect(() => {
    const configToFind = autoRunConfig || pendingAutoRun;
    
    if (configToFind && configurations.length > 0) {
      // Find the config in our list - prioritize exact ID match over flow_name match
      // This prevents running an older config with the same flow_name when multiple
      // configurations share the same flow (e.g. clone, or creating a new config for the same flow)
      const configToRun = 
        configurations.find(c => c.id === configToFind.id) || 
        configurations.find(c => c.flow_name === configToFind.flow_name);
      
      if (configToRun) {
        
        // Set selectedConfigs FIRST (before any other state updates)
        setSelectedConfigs([configToRun.id]);
        
        // Set the config to auto-expand (opens terminal view)
        setAutoExpandConfigId(configToRun.id);
        
        // Small delay to ensure state is updated before triggering batch generate
        setTimeout(() => {
          setBatchGenerateTrigger(prev => prev + 1);
        }, 150);
        
        // Clear pending and autoRunConfig
        setPendingAutoRun(null);
        if (onAutoRunCompleteRef.current) {
          onAutoRunCompleteRef.current();
        }
      } else if (autoRunConfig) {
        // Config not found yet, save it as pending and wait for next load
        setPendingAutoRun(autoRunConfig);
        if (onAutoRunCompleteRef.current) {
          onAutoRunCompleteRef.current();
        }
        // Trigger a reload of configurations
        if (loadConfigurationsRef.current) {
          loadConfigurationsRef.current();
        }
      }
    }
  }, [autoRunConfig, configurations, pendingAutoRun]);

  /**
   * Update execution state for a config
   */
  const updateConfigExecutionState = (configId, updates) => {
    // Use the prop from parent to update execution state
    if (typeof updates === 'function') {
      const currentState = executionStates[configId] || {};
      onUpdateExecutionState(configId, updates(currentState));
    } else {
      onUpdateExecutionState(configId, updates);
    }
  };

  /**
   * Get execution state for a config
   */
  const getExecutionStateForConfig = (configId) => {
    return executionStates[configId] || null;
  };

  /**
   * Handle adding a new configuration (after wizard completion)
   */
  const handleConfigurationAdded = (newConfig) => {
    // Refresh configurations list
    loadConfigurations();
  };

  /**
   * Handle deleting a configuration or draft
   */
  const handleDeleteConfiguration = async (configId) => {
    try {
      // Check if it's a draft
      const config = configurations.find(c => c.id === configId);
      
      if (config && config.isDraft) {
        // Delete draft from localStorage
        const draftsJson = localStorage.getItem('wizard_drafts');
        if (draftsJson) {
          const drafts = JSON.parse(draftsJson);
          const updatedDrafts = drafts.filter(d => d.id !== configId);
          localStorage.setItem('wizard_drafts', JSON.stringify(updatedDrafts));
        }
        
        // Remove from configurations list
        setConfigurations(prev => prev.filter(c => c.id !== configId));
      } else {
        // Delete from backend (regular configuration)
        await savedConfigAPI.delete(configId);
        setConfigurations(prev => prev.filter(c => c.id !== configId));
      }
    } catch (err) {
      console.error('Error deleting configuration:', err);
      setError('Failed to delete configuration: ' + err.message);
    }
  };

  /**
   * Determine which step to start at when editing
   * For "Not Configured" flows, navigate to the step where the user left off:
   * - No model → start at model-configuration
   * - Has model but no dataset → start at dataset-configuration
   * For fully configured flows, use existing logic
   */
  const getEditStartStep = (config) => {
    const isDraft = config.isDraft;
    
    // Multiple ways to detect custom flows:
    // 1. Flow name contains "(Custom)" or "(Copy)"
    // 2. Flow path contains "custom_flows"
    // 3. Explicit isCustomFlow flag
    const hasCustomSuffix = config.flow_name?.includes('(Custom)');
    const hasCopySuffix = config.flow_name?.includes('(Copy)'); // Templates loaded get "(Copy)" suffix
    const hasCustomPath = config.flow_path?.includes('custom_flows');
    const hasCustomFlag = config.isCustomFlow === true;
    const isCustomFlow = hasCustomSuffix || hasCopySuffix || hasCustomPath || hasCustomFlag;
    
    // Check what's configured
    const modelConfig = config.model_configuration || config.model_config || {};
    const datasetConfig = config.dataset_configuration || config.dataset_config || {};
    const hasModel = modelConfig.model && modelConfig.model !== 'Not configured' && modelConfig.model !== 'Not specified';
    const hasDataset = datasetConfig.data_files && datasetConfig.data_files !== 'Not configured' && datasetConfig.data_files !== 'Not specified';
    
    // For DRAFTS - ALWAYS start at Build Flow
    if (isDraft) {
      return { 
        stepName: 'build-flow',
        sourceType: 'draft',
        message: 'Resuming draft at Build Flow'
      };
    }
    
    // For CUSTOM FLOWS (created with blank format or clone existing flow) - start at Build Flow
    if (isCustomFlow) {
      return { 
        stepName: 'build-flow',
        sourceType: 'clone', // Use clone mode to enable Build Flow step
        message: 'Editing custom flow at Build Flow'
      };
    }
    
    // For NOT CONFIGURED flows - start at the step where they need to continue
    if (config.status === 'not_configured') {
      if (!hasModel) {
        // No model configured - start at model configuration
        return { 
          stepName: 'model-configuration',
          sourceType: 'existing',
          message: 'Continue configuring model'
        };
      }
      if (!hasDataset) {
        // Has model but no dataset - start at dataset configuration
        return { 
          stepName: 'dataset-configuration',
          sourceType: 'existing',
          message: 'Continue configuring dataset'
        };
      }
      // Both configured but still "not_configured" status - go to review
      return { 
        stepName: 'review',
        sourceType: 'existing',
        message: 'Review and save configuration'
      };
    }
    
    // For EXISTING FLOWS (fully configured) - go directly to Select Flow step (skip source selection)
    return { 
      stepName: 'select-existing',
      sourceType: 'existing',
      message: 'Editing existing flow configuration'
    };
  };
  
  /**
   * Handle editing a configuration or resuming a draft
   */
  const handleEditConfiguration = (config) => {
    // Determine where to start in the wizard
    const startStep = getEditStartStep(config);
    
    
    // Check if it's a draft flow
    if (config.isDraft) {
      // Resume draft in wizard
      if (onNavigate) {
        onNavigate('configure-flow', { 
          resumeDraft: true,
          draftData: config.draftData,
          startStepName: startStep.stepName,
          sourceType: startStep.sourceType
        });
      }
    } else {
      // Edit existing configuration - navigate with pre-filled data
      if (onEditConfiguration) {
        onEditConfiguration(config, {
          startStepName: startStep.stepName,
          sourceType: startStep.sourceType
        });
      }
    }
  };

  /**
   * Handle cloning a configuration - opens wizard with copied config as new
   * For Clone & Modify, we skip the source selection step:
   * - Existing flows → go directly to Select Flow step
   * - Custom/Blank/Clone flows → go directly to Build Flow step
   */
  const handleCloneConfiguration = (config) => {
    
    // Multiple ways to detect custom flows
    const hasCustomSuffix = config.flow_name?.includes('(Custom)');
    const hasCopySuffix = config.flow_name?.includes('(Copy)');
    const hasCustomPath = config.flow_path?.includes('custom_flows');
    const hasCustomFlag = config.isCustomFlow === true;
    const isCustomFlow = hasCustomSuffix || hasCopySuffix || hasCustomPath || hasCustomFlag;
    
    // Create a cloned configuration object
    const clonedConfig = {
      ...config,
      id: undefined, // Clear ID so it creates a new one
      flow_name: `${config.flow_name} (Copy)`,
      isClone: true,
      isCustomFlow: isCustomFlow, // Preserve the custom flow flag
    };
    
    // Determine start step based on flow type (skip source selection)
    let startStepName, sourceType;
    
    if (isCustomFlow) {
      // Custom flows go to Build Flow step
      startStepName = 'build-flow';
      sourceType = 'clone';
    } else {
      // Existing flows go to Select Flow step
      startStepName = 'select-existing';
      sourceType = 'existing';
    }
    
    
    // Navigate to wizard with cloned data
    if (onNavigate) {
      onNavigate('configure-flow', { 
        clonedConfig: clonedConfig,
        startStepName: startStepName,
        sourceType: sourceType,
        isCloning: true
      });
    }
  };

  /**
   * Get dataset name from config
   */
  const getDatasetName = (config) => {
    const datasetConfig = config.dataset_configuration || config.dataset_config;
    if (datasetConfig?.data_files) {
      const path = datasetConfig.data_files;
      const parts = path.split('/');
      return parts[parts.length - 1];
    }
    return '';
  };

  /**
   * Get autocomplete suggestions based on search type and input
   */
  const getAutocompleteSuggestions = (type, input) => {
    if (!input || input.length < 1) return [];
    
    const inputLower = input.toLowerCase();
    const uniqueValues = new Set();
    
    configurations.forEach(config => {
      const modelConfig = config.model_configuration || config.model_config || {};
      
      switch (type) {
        case 'flow_name':
          if (config.flow_name?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(config.flow_name);
          }
          break;
        case 'model':
          if (modelConfig.model?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(modelConfig.model);
          }
          break;
        case 'dataset': {
          const datasetName = getDatasetName(config);
          if (datasetName?.toLowerCase().includes(inputLower)) {
            uniqueValues.add(datasetName);
          }
          break;
        }
        case 'tags':
          config.tags?.forEach(tag => {
            if (tag.toLowerCase().includes(inputLower)) {
              uniqueValues.add(tag);
            }
          });
          break;
        default:
          break;
      }
    });
    
    return Array.from(uniqueValues).slice(0, 10); // Limit to 10 suggestions
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
  }, [searchValue, searchType, configurations]);

  /**
   * Add a search tag
   */
  const addSearchTag = (type, value) => {
    // Check if tag already exists
    const exists = searchTags.some(tag => tag.type === type && tag.value === value);
    if (!exists) {
      setSearchTags([...searchTags, { type, value }]);
    }
    setSearchValue(''); // Clear search input
    setShowAutocomplete(false);
  };

  /**
   * Remove a search tag
   */
  const removeSearchTag = (index) => {
    setSearchTags(searchTags.filter((_, i) => i !== index));
  };

  /**
   * Filter configurations based on search tags
   */
  const filteredConfigurations = configurations.filter(config => {
    // If no tags, show all
    if (searchTags.length === 0) return true;
    
    // All tags must match (AND logic)
    return searchTags.every(tag => {
    const modelConfig = config.model_configuration || config.model_config || {};
    
      switch (tag.type) {
      case 'flow_name':
          return config.flow_name === tag.value;
      case 'model':
          return modelConfig.model === tag.value;
      case 'dataset':
          return getDatasetName(config) === tag.value;
      case 'tags':
          return config.tags?.includes(tag.value);
      default:
          return true;
    }
    });
  });


  /**
   * Render empty state (no configurations yet)
   */
  const renderEmptyState = () => (
    <EmptyState>
      <EmptyStateHeader
        titleText="No configurations yet"
        icon={<EmptyStateIcon icon={CubesIcon} />}
        headingLevel="h1"
      />
      <EmptyStateBody>
        Get started by creating your first flow configuration. You can use an existing flow 
        from the SDG Hub library or build a custom flow from scratch.
      </EmptyStateBody>
      <EmptyStateFooter>
        <Button
          variant="primary"
          onClick={() => {
            if (onNavigate) {
              // Clear wizard session to start fresh
              sessionStorage.removeItem('wizard_session_state');
              onNavigate('configure-flow', null);
            }
          }}
        >
          Configure Flow
        </Button>
      </EmptyStateFooter>
    </EmptyState>
  );

  /**
   * Get search placeholder text based on search type
   */
  const getSearchPlaceholder = () => {
    switch (searchType) {
      case 'flow_name':
        return 'Search by flow name';
      case 'model':
        return 'Search by model';
      case 'dataset':
        return 'Search by dataset name';
      case 'tags':
        return 'Search by tags';
      default:
        return 'Search configurations';
    }
  };

  /**
   * Render toolbar with search and action buttons
   */
  /**
   * Stop selected running flows
   */
  const handleStopSelected = async () => {
    try {
      for (const configId of selectedConfigs) {
        const state = executionStates[configId];
        if (!state || !state.isRunning) {
          continue;
        }
        
        await executionAPI.cancel(configId);
        
        if (state.eventSource) {
          state.eventSource.close();
        }
        
        onUpdateExecutionState(configId, {
          isRunning: false,
          status: 'cancelled',
          rawOutput: (state.rawOutput || '') + '\n⚠️ Generation cancelled by user\n',
          eventSource: null
        });
        
        if (state.runId) {
          runsAPI.update(state.runId, {
            status: 'cancelled',
            error_message: 'Cancelled by user'
          });
        }
      }
      
      setIsActionsMenuOpen(false);
    } catch (error) {
      console.error('Error stopping flows:', error);
      setError('Failed to stop flows: ' + error.message);
    }
  };

  const renderToolbar = () => (
    <Toolbar id="configurations-toolbar">
      <ToolbarContent>
        <ToolbarItem>
          <select
            value={searchType}
            onChange={(e) => {
              setSearchType(e.target.value);
              setSearchValue(''); // Clear search when changing type
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
            <option value="dataset">Dataset</option>
            <option value="tags">Tags</option>
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
        <ToolbarItem variant="separator" />
        <ToolbarItem>
          <Dropdown
            isOpen={isActionsMenuOpen}
            onSelect={() => setIsActionsMenuOpen(false)}
            onOpenChange={(isOpen) => setIsActionsMenuOpen(isOpen)}
            toggle={(toggleRef) => (
              <MenuToggle
                ref={toggleRef}
                onClick={() => setIsActionsMenuOpen(!isActionsMenuOpen)}
                variant="secondary"
              >
                Actions
              </MenuToggle>
            )}
          >
            <DropdownList>
              {(() => {
                // Check if any selected config is not configured
                const hasUnconfiguredSelected = selectedConfigs.some(configId => {
                  const config = configurations.find(c => c.id === configId);
                  return config?.status === 'not_configured';
                });
                
                // Check if any selected config is running
                const hasRunningSelected = selectedConfigs.some(configId => {
                  const state = executionStates[configId];
                  return state?.isRunning;
                });
                
                const isRunDisabled = selectedConfigs.length === 0 || hasUnconfiguredSelected;
                const isStopDisabled = selectedConfigs.length === 0 || !hasRunningSelected;
                
                return (
                  <>
                    <Tooltip
                      content={
                        selectedConfigs.length === 0 
                          ? "Select configurations to run" 
                          : hasUnconfiguredSelected 
                            ? "You have unconfigured flows selected" 
                            : ""
                      }
                      trigger={isRunDisabled ? "mouseenter" : "manual"}
                    >
                      <DropdownItem
                        key="run"
                        icon={<PlayIcon />}
                        onClick={() => {
                          if (isRunDisabled) return;
                          setBatchGenerateTrigger(prev => prev + 1);
                          setIsActionsMenuOpen(false);
                        }}
                        isDisabled={isRunDisabled}
                      >
                        Run ({selectedConfigs.length})
                      </DropdownItem>
                    </Tooltip>
                    <Tooltip
                      content={
                        selectedConfigs.length === 0 
                          ? "Select configurations to stop" 
                          : !hasRunningSelected 
                            ? "No running flows selected" 
                            : ""
                      }
                      trigger={isStopDisabled ? "mouseenter" : "manual"}
                    >
                      <DropdownItem
                        key="stop"
                        icon={<StopIcon />}
                        onClick={() => {
                          if (isStopDisabled) return;
                          handleStopSelected();
                        }}
                        isDisabled={isStopDisabled}
                      >
                        Stop ({selectedConfigs.filter(id => executionStates[id]?.isRunning).length})
                      </DropdownItem>
                    </Tooltip>
                    <DropdownItem
                      key="delete"
                      icon={<TrashIcon />}
                      onClick={async () => {
                        if (selectedConfigs.length === 0) {
                          setError('Please select at least one configuration to delete');
                          return;
                        }
                        
                        const confirmed = window.confirm(
                          `Are you sure you want to delete ${selectedConfigs.length} configuration${selectedConfigs.length > 1 ? 's' : ''}?\n\nThis action cannot be undone.`
                        );
                        
                        if (!confirmed) {
                          setIsActionsMenuOpen(false);
                          return;
                        }
                        
                        // Delete all selected configurations
                        try {
                          for (const configId of selectedConfigs) {
                            await handleDeleteConfiguration(configId);
                          }
                          setSelectedConfigs([]); // Clear selection after deletion
                          setIsActionsMenuOpen(false);
                        } catch (err) {
                          console.error('Error deleting configurations:', err);
                          setError('Failed to delete some configurations: ' + err.message);
                        }
                      }}
                      isDisabled={selectedConfigs.length === 0}
                      isDanger
                    >
                      Delete ({selectedConfigs.length})
                    </DropdownItem>
                  </>
                );
              })()}
            </DropdownList>
          </Dropdown>
        </ToolbarItem>
        <ToolbarItem>
          <Button
            variant="secondary"
            icon={<PlusCircleIcon />}
            onClick={() => {
              if (onNavigate) {
                // Clear wizard session to start fresh
                sessionStorage.removeItem('wizard_session_state');
                onNavigate('configure-flow', null);
              }
            }}
          >
            Configure Flow
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
                  <strong>{tag.type === 'flow_name' ? 'Flow' : tag.type === 'dataset' ? 'Dataset' : tag.type.charAt(0).toUpperCase() + tag.type.slice(1)}:</strong> {tag.value}
                </Chip>
              ))}
            </ChipGroup>
          </ToolbarItem>
        </ToolbarContent>
      )}
    </Toolbar>
  );

  /**
   * Render loading state
   */
  if (isLoading) {
    return (
      <PageSection isCenterAligned>
        <Spinner size="xl" />
      </PageSection>
    );
  }

  /**
   * Render summary section - reflects current configuration and execution statuses
   * Split into 2 rows: Configuration Status (top) and Execution Status (bottom)
   */
  const renderSummary = () => {
    // Count based on configuration statuses in the table
    const configuredCount = configurations.filter(c => c.status === 'configured').length;
    const notConfiguredCount = configurations.filter(c => c.status === 'not_configured').length;
    const draftCount = configurations.filter(c => c.status === 'draft' || c.isDraft).length;
    
    // Count execution states (only for configs in the table)
    const runningCount = configurations.filter(c => {
      const state = executionStates[c.id];
      return state?.isRunning;
    }).length;
    
    const failedCount = configurations.filter(c => {
      const state = executionStates[c.id];
      return state?.status === 'failed' || state?.status === 'error';
    }).length;
    
    const completedCount = configurations.filter(c => {
      const state = executionStates[c.id];
      return state?.status === 'completed';
    }).length;
    
    const stoppedCount = configurations.filter(c => {
      const state = executionStates[c.id];
      return state?.status === 'cancelled' || state?.status === 'stopped';
    }).length;
    
    return (
      <div style={{ marginBottom: '24px' }}>
        {/* Row 1: Configuration Status */}
        <div style={{ marginBottom: '8px' }}>
          <span style={{ 
            fontSize: '12px', 
            fontWeight: 600, 
            color: 'var(--pf-v5-global--Color--200)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}>
            Configuration Status
          </span>
        </div>
        <Grid hasGutter style={{ marginBottom: '16px' }}>
          <GridItem sm={12} md={4} lg={4}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <CheckCircleIcon style={{ color: 'var(--pf-v5-global--success-color--100)' }} />
                  <span>Configured</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{configuredCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
          
          <GridItem sm={12} md={4} lg={4}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <ExclamationCircleIcon style={{ color: 'var(--pf-v5-global--warning-color--100)' }} />
                  <span>Not Configured</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{notConfiguredCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
          
          <GridItem sm={12} md={4} lg={4}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <CubesIcon style={{ color: 'var(--pf-v5-global--Color--200)' }} />
                  <span>Drafts</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{draftCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
        </Grid>
        
        {/* Row 2: Execution Status */}
        <div style={{ marginBottom: '8px' }}>
          <span style={{ 
            fontSize: '12px', 
            fontWeight: 600, 
            color: 'var(--pf-v5-global--Color--200)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}>
            Execution Status
          </span>
        </div>
        <Grid hasGutter>
          <GridItem sm={12} md={3} lg={3}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <InProgressIcon style={{ color: 'var(--pf-v5-global--info-color--100)' }} />
                  <span>Running</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{runningCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
          
          <GridItem sm={12} md={3} lg={3}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <ExclamationCircleIcon style={{ color: 'var(--pf-v5-global--danger-color--100)' }} />
                  <span>Failed</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{failedCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
          
          <GridItem sm={12} md={3} lg={3}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <CheckCircleIcon style={{ color: 'var(--pf-v5-global--palette--green-500)' }} />
                  <span>Completed</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{completedCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
          
          <GridItem sm={12} md={3} lg={3}>
            <Card isCompact isFullHeight>
              <CardTitle>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                  <StopIcon style={{ color: 'var(--pf-v5-global--warning-color--100)' }} />
                  <span>Stopped</span>
                </div>
              </CardTitle>
              <CardBody>
                <Title headingLevel="h2" size="3xl">{stoppedCount}</Title>
              </CardBody>
            </Card>
          </GridItem>
        </Grid>
      </div>
    );
  };

  return (
    <>
      {/* Page Header */}
      <PageSection variant="light">
        <Title headingLevel="h1" size="2xl">Data Generation Flows</Title>
        <p style={{ 
          marginTop: '8px', 
          color: 'var(--pf-v5-global--Color--200)',
          fontSize: '14px'
        }}>
          Configure and manage synthetic data generation flows. Create new flow configurations 
          using existing flows or build custom flows from scratch.
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

        {/* Summary Section - only show when there are configurations */}
        {configurations.length > 0 && renderSummary()}

        {configurations.length === 0 ? (
          renderEmptyState()
        ) : (
          <ConfigurationList
            configurations={filteredConfigurations}
            onDelete={handleDeleteConfiguration}
            onRefresh={loadConfigurations}
            onEdit={handleEditConfiguration}
            onClone={handleCloneConfiguration}
            selectedConfigs={selectedConfigs}
            onSelectedConfigsChange={setSelectedConfigs}
            batchGenerateTrigger={batchGenerateTrigger}
            executionStates={executionStates}
            onUpdateExecutionState={onUpdateExecutionState}
            getExecutionState={getExecutionState}
            renderToolbar={renderToolbar}
            autoExpandConfigId={autoExpandConfigId}
            onAutoExpandComplete={() => setAutoExpandConfigId(null)}
          />
        )}
      </PageSection>
    </>
  );
};

export default DataGenerationFlowsPage;

