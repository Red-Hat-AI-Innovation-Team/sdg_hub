import React, { useState, useEffect, useRef } from 'react';
import ConfigurationDetailView from './ConfigurationDetailView';
import ConfigurationTable from './ConfigurationTable';
import api, { savedConfigAPI, executionAPI, runsAPI, checkpointAPI, API_BASE_URL } from '../../services/api';

/**
 * List of flow configurations
 */
const STREAM_BASE_URL = API_BASE_URL.replace(/\/$/, '');

// Session storage key for expanded config persistence
const EXPANDED_CONFIG_KEY = 'flows_expanded_config_id';

const ConfigurationList = ({ 
  configurations, 
  onDelete, 
  onRefresh, 
  onEdit,
  onClone,
  selectedConfigs: parentSelectedConfigs, 
  onSelectedConfigsChange, 
  batchGenerateTrigger,
  executionStates: parentExecutionStates,
  onUpdateExecutionState,
  getExecutionState: parentGetExecutionState,
  renderToolbar,
  autoExpandConfigId,
  onAutoExpandComplete,
}) => {
  const [selectedConfigs, setSelectedConfigs] = useState([]);
  
  // Track previous batch trigger to avoid re-running on selection changes
  const prevBatchTriggerRef = useRef(0);
  
  // Checkpoint info for expanded config
  const [checkpointInfo, setCheckpointInfo] = useState(null);
  
  // Initialize expandedConfig from sessionStorage if available
  const [expandedConfig, setExpandedConfig] = useState(() => {
    try {
      const savedExpandedId = sessionStorage.getItem(EXPANDED_CONFIG_KEY);
      if (savedExpandedId && configurations.length > 0) {
        const found = configurations.find(c => c.id === savedExpandedId);
        return found || null;
      }
    } catch (error) {
      console.error('Failed to load expanded config from sessionStorage:', error);
    }
    return null;
  });
  
  // Use parent execution states if provided, otherwise create local
  const executionStates = parentExecutionStates || {};
  
  /**
   * Update execution state - uses parent's update function if available
   */
  const updateConfigExecutionState = (configId, updates) => {
    if (onUpdateExecutionState) {
      onUpdateExecutionState(configId, updates);
    }
  };
  
  // Legacy single execution state for backward compatibility
  const [executionState, setExecutionState] = useState({
    configId: null,
    type: null, // 'dry_run' or 'generate'
    isRunning: false,
    logs: [],
    rawOutput: '',
    result: null,
    eventSource: null,
  });

  /**
   * Sync local selected configs with parent
   */
  useEffect(() => {
    if (parentSelectedConfigs !== undefined) {
      setSelectedConfigs(parentSelectedConfigs);
    }
  }, [parentSelectedConfigs]);

  /**
   * Auto-expand config when autoExpandConfigId is set (after Save and Run)
   */
  useEffect(() => {
    if (autoExpandConfigId && configurations.length > 0) {
      const configToExpand = configurations.find(c => c.id === autoExpandConfigId);
      if (configToExpand) {
        // Immediately expand to show terminal view
        setExpandedConfig(configToExpand);
        // Clear the auto-expand flag after a short delay
        setTimeout(() => {
          if (onAutoExpandComplete) {
            onAutoExpandComplete();
          }
        }, 200);
      }
    }
  }, [autoExpandConfigId, configurations]);
  
  /**
   * Persist expanded config to sessionStorage whenever it changes
   */
  useEffect(() => {
    try {
      if (expandedConfig?.id) {
        sessionStorage.setItem(EXPANDED_CONFIG_KEY, expandedConfig.id);
      } else {
        sessionStorage.removeItem(EXPANDED_CONFIG_KEY);
      }
    } catch (error) {
      console.error('Failed to save expanded config to sessionStorage:', error);
    }
  }, [expandedConfig]);
  
  /**
   * Restore expanded config when configurations are loaded (for refresh/navigation scenarios)
   */
  useEffect(() => {
    if (configurations.length > 0 && !expandedConfig && !autoExpandConfigId) {
      try {
        const savedExpandedId = sessionStorage.getItem(EXPANDED_CONFIG_KEY);
        if (savedExpandedId) {
          const found = configurations.find(c => c.id === savedExpandedId);
          if (found) {
            setExpandedConfig(found);
          } else {
            // Config no longer exists, clear the saved state
            sessionStorage.removeItem(EXPANDED_CONFIG_KEY);
          }
        }
      } catch (error) {
        console.error('Failed to restore expanded config:', error);
      }
    }
  }, [configurations]);

  /**
   * Fetch checkpoint info when expanded config is in a resumable state
   */
  useEffect(() => {
    const fetchCheckpointInfo = async () => {
      if (!expandedConfig?.id) {
        setCheckpointInfo(null);
        return;
      }
      
      const state = getExecutionStateForConfig(expandedConfig.id);
      const isResumable = state?.status === 'failed' || 
                          state?.status === 'error' || 
                          state?.status === 'cancelled' || 
                          state?.status === 'stopped';
      
      if (isResumable) {
        try {
          const info = await checkpointAPI.getCheckpointInfo(expandedConfig.id);
          setCheckpointInfo(info);
        } catch (error) {
          console.error('Failed to fetch checkpoint info:', error);
          setCheckpointInfo(null);
        }
      } else {
        setCheckpointInfo(null);
      }
    };
    
    fetchCheckpointInfo();
  }, [expandedConfig?.id, executionStates]);

  /**
   * Check if configuration is currently running
   */
  const isConfigRunning = (configId) => {
    // Check both new multi-state and legacy single state
    return (
      (executionStates[configId]?.isRunning) ||
      (executionState.configId === configId && executionState.isRunning)
    );
  };

  /**
   * Get execution state for a config (checks both new and legacy state)
   */
  const getExecutionStateForConfig = (configId) => {
    if (executionStates[configId]) {
      return executionStates[configId];
    }
    if (executionState.configId === configId) {
      return executionState;
    }
    return null;
  };

  /**
   * Clear terminal output for a configuration
   */
  const handleClearTerminal = (configId) => {
    updateConfigExecutionState(configId, (prevState) => ({
      ...prevState,
      rawOutput: '',
      result: null,
    }));
  };

  /**
   * Handle stop execution
   */
  const handleStop = async (config) => {
    const configState = getExecutionStateForConfig(config.id);
    
    // 1. Store the EventSource reference before clearing it
    const eventSourceToClose = configState?.eventSource;
    
    // 2. IMMEDIATELY mark as cancelled to prevent race conditions
    updateConfigExecutionState(config.id, (prevState) => ({
      ...prevState,
      isRunning: false,
      status: 'cancelled',
      rawOutput: (prevState?.rawOutput || '') + '\n⚠️ Generation cancelled by user\n',
      eventSource: null, // Clear the reference immediately
      generationId: null, // Clear the generation ID
      cancelledAt: Date.now() // Add timestamp for cancellation
    }));
    
    // 3. Close frontend connection (after state update to ensure handlers see cancelled state)
    if (eventSourceToClose) {
      try {
        eventSourceToClose.close();
      } catch (e) {
        console.error('Error closing EventSource:', e);
      }
    }
    
    // 4. Call backend to stop process
    try {
      await executionAPI.cancel(config.id);
      
      // Also update run record status to cancelled (not failed)
      if (configState?.runId) {
        await runsAPI.update(configState.runId, {
          status: 'cancelled',
          error_message: 'Cancelled by user'
        });
      }
    } catch (error) {
      console.error('Error cancelling generation:', error);
    }
  };

  /**
   * Toggle configuration selection
   */
  const toggleSelection = (configId) => {
    const newSelection = selectedConfigs.includes(configId)
      ? selectedConfigs.filter(id => id !== configId)
      : [...selectedConfigs, configId];
    
    setSelectedConfigs(newSelection);
    if (onSelectedConfigsChange) {
      onSelectedConfigsChange(newSelection);
    }
  };

  /**
   * Handle clicking on flow name to expand details
   */
  const handleFlowNameClick = (config) => {
    // If clicking on a running config, open detail view with terminal
    const isRunning = isConfigRunning(config.id);
    if (isRunning) {
      setExpandedConfig(config);
    } else {
      setExpandedConfig(expandedConfig?.id === config.id ? null : config);
    }
  };

  /**
   * Handle delete
   */
  const handleDelete = async (configId) => {
    if (!window.confirm('Are you sure you want to delete this configuration?')) {
      return;
    }

    // Check if running and stop it first
    if (isConfigRunning(configId)) {
      await handleStop({ id: configId });
    }

    await onDelete(configId);
  };

  /**
   * Handle dry run execution with streaming
   */
  const handleDryRun = async (config) => {
    let eventSource = null;
    
    try {
      // Open detail view first and switch to terminal tab
      setExpandedConfig(config);
      
      // Initialize execution state for this config in the map
      updateConfigExecutionState(config.id, {
        configId: config.id,
        type: 'dry_run',
        isRunning: true,
        logs: [],
        rawOutput: '',
        result: null,
        eventSource: null,
      });

      // Step 1: Load configuration
      await savedConfigAPI.load(config.id);

      // Step 2: Load dataset
      const datasetConfig = config.dataset_configuration || config.dataset_config;
      if (datasetConfig && datasetConfig.data_files && datasetConfig.data_files !== '.') {
        await api.post('/api/dataset/load', datasetConfig);
      }

      // Step 3: Run dry run with streaming using saved dry run configuration
      const dryRunSettings = config.dry_run_configuration || {
        sample_size: 2,
        enable_time_estimation: true,
        max_concurrency: 10
      };
      
      const params = new URLSearchParams({
        sample_size: dryRunSettings.sample_size,
        enable_time_estimation: dryRunSettings.enable_time_estimation,
        max_concurrency: dryRunSettings.max_concurrency,
      });
      
      const url = `${STREAM_BASE_URL}/api/flow/dry-run-stream?${params}`;
      eventSource = new EventSource(url);
      
      updateConfigExecutionState(config.id, {
        ...(executionStates[config.id] || {}),
        eventSource,
      });
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'start') {
            updateConfigExecutionState(config.id, (prevState) => {
              const newOutput = (prevState.rawOutput || '') + data.message + '\n';
              return {
                ...prevState,
                rawOutput: newOutput,
              };
            });
          } else if (data.type === 'log') {
            updateConfigExecutionState(config.id, (prevState) => {
              const newOutput = (prevState.rawOutput || '') + data.message + '\n';
              return {
                ...prevState,
                rawOutput: newOutput,
              };
            });
          } else if (data.type === 'complete') {
            updateConfigExecutionState(config.id, (prevState) => {
              const newOutput = (prevState.rawOutput || '') + `\n✅ Dry run completed in ${data.result?.execution_time_seconds?.toFixed(2)}s\n`;
              eventSource.close();
              return {
                ...prevState,
                isRunning: false,
                rawOutput: newOutput,
                result: data.result,
                eventSource: null,
              };
            });
          } else if (data.type === 'error') {
            updateConfigExecutionState(config.id, (prevState) => {
              const newOutput = (prevState.rawOutput || '') + `\n❌ Error: ${data.message}\n`;
              eventSource.close();
              return {
                ...prevState,
                isRunning: false,
                rawOutput: newOutput,
                eventSource: null,
              };
            });
          }
        } catch (err) {
          console.error('Error parsing event:', err);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        updateConfigExecutionState(config.id, (prevState) => ({
          ...prevState,
          isRunning: false,
          rawOutput: (prevState.rawOutput || '') + '\n❌ Connection to server lost\n',
          eventSource: null,
        }));
        eventSource.close();
      };

    } catch (error) {
      console.error('Dry run error:', error);
      if (eventSource) {
        eventSource.close();
      }
      updateConfigExecutionState(config.id, (prevState) => ({
        ...(prevState || {}),
        configId: config.id,
        type: 'dry_run',
        isRunning: false,
        rawOutput: (prevState?.rawOutput || '') + `\n❌ Error: ${error.response?.data?.detail || error.message}\n`,
        result: null,
        eventSource: null,
      }));
    }
  };

  /**
   * Handle batch generation triggered from parent
   * Only runs when batchGenerateTrigger actually increases (not on selection changes)
   */
  React.useEffect(() => {
    // Only run if trigger actually changed (increased)
    if (batchGenerateTrigger > prevBatchTriggerRef.current) {
      prevBatchTriggerRef.current = batchGenerateTrigger;
      
      // Use parentSelectedConfigs directly to avoid race condition with local state sync
      const effectiveSelectedConfigs = parentSelectedConfigs || selectedConfigs;
      
      if (effectiveSelectedConfigs.length > 0) {
        const configsToGenerate = configurations.filter(c => effectiveSelectedConfigs.includes(c.id));
        
        // Start all generations concurrently
        configsToGenerate.forEach(config => {
          handleGenerateForConfig(config);
        });
      }
    }
  }, [batchGenerateTrigger]);

  /**
   * Handle generate execution for a specific config
   * @param {Object} config - The configuration to run
   * @param {boolean} resumeFromCheckpoint - If true, resume from existing checkpoints
   */
  const handleGenerateForConfig = async (config, resumeFromCheckpoint = false) => {
    let eventSource = null;
    const runId = `run_${config.id}_${Date.now()}`;
    const startTime = new Date().toISOString();
    
    try {
      // Don't open detail view automatically - stay on flows page
      // User can click the flow name to see terminal if they want
      
      // Create run record
      const modelConfig = config.model_configuration || config.model_config || {};
      const datasetConfig = config.dataset_configuration || config.dataset_config || {};
      
      const runRecord = {
        run_id: runId,
        config_id: config.id,
        flow_name: config.flow_name,
        flow_type: config.flow_id?.startsWith('custom-') ? 'custom' : 'existing',
        model_name: modelConfig.model || 'Unknown',
        status: 'running',
        start_time: startTime,
        input_samples: datasetConfig.num_samples || 0,
        dataset_file: datasetConfig.data_files || null,
      };
      
      await runsAPI.create(runRecord);
      
      // Initialize execution state for this config
      // COMPLETELY CLEAR any previous state to start fresh
      updateConfigExecutionState(config.id, {
        configId: config.id,
        configName: config.flow_name,
        flowName: config.flow_name,
        type: 'generate',
        status: 'running', // Explicitly set status
        isRunning: true,
        logs: [], // Clear all previous logs
        rawOutput: '🚀 Starting generation...\n',
        result: null,
        eventSource: null,
        runId: runId,
        startTime: startTime,
        cancelledAt: null,
        completedAt: null, // Clear previous completion timestamp
        error: null,
        duration: null,
        outputSamples: null,
        generationId: null,
        isRestarting: true, // Flag to prevent false notifications from old state
      });

      // Check if there's an existing EventSource and close it just in case
      const existingState = getExecutionStateForConfig(config.id);
      if (existingState && existingState.eventSource) {
        try {
          existingState.eventSource.close();
        } catch (e) {
          console.error('Error closing existing EventSource:', e);
        }
      }

      // Wait a moment to ensure the old EventSource is fully closed
      await new Promise(resolve => setTimeout(resolve, 100));

      // NOTE: We no longer call savedConfigAPI.load() or api.post('/api/dataset/load') here
      // because the backend's generate-stream endpoint now loads the configuration directly
      // from saved_configurations using the config_id parameter. This prevents race conditions
      // when running multiple flows in parallel (batch runs).

      // Add resume message if applicable
      const resumeMsg = resumeFromCheckpoint ? '📂 Resuming from checkpoint...\n\n' : '';
      if (resumeMsg) {
        updateConfigExecutionState(config.id, (prevState) => ({
          ...prevState,
          rawOutput: prevState.rawOutput + resumeMsg,
        }));
      }

      // Start generation with streaming
      const maxConcurrency = 100;
      // Get save_freq from model config additional_params, or use default of 10
      const saveFreq = modelConfig?.additional_params?.save_freq || 10;
      const url = `${STREAM_BASE_URL}/api/flow/generate-stream?config_id=${encodeURIComponent(config.id)}&max_concurrency=${maxConcurrency}&enable_checkpoints=true&save_freq=${saveFreq}&resume_from_checkpoint=${resumeFromCheckpoint}`;
      
      eventSource = new EventSource(url);
      
      // Store the new event source in state immediately to prevent race conditions
      updateConfigExecutionState(config.id, (prevState) => ({
        ...prevState,
        eventSource,
      }));
      
      // Create a unique generation ID for this run to prevent stale events
      const generationId = `${config.id}_${Date.now()}`;
      
      // Store generation ID in state - make sure to clear cancelled status
      updateConfigExecutionState(config.id, (prevState) => ({
        ...prevState,
        generationId,
        status: 'running', // Explicitly set to running
        isRunning: true,
        cancelledAt: null // Clear any previous cancellation
      }));
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'start') {
            updateConfigExecutionState(config.id, (prevState) => {
              // Check if this event is for the current generation
              if (prevState.generationId !== generationId) {
                try { eventSource.close(); } catch(e) {}
                return prevState;
              }
              
              // Check if this EventSource is still the active one
              if (prevState.eventSource !== eventSource) {
                try { eventSource.close(); } catch(e) {}
                return prevState;
              }
              
              // Ensure we are still running (if user cancelled quickly, this might be stale)
              if (!prevState.isRunning || prevState.status === 'cancelled') {
                try { eventSource.close(); } catch(e) {}
                return prevState;
              }
              
              const newOutput = (prevState.rawOutput || '') + data.message + '\n';
              return {
                ...prevState,
                rawOutput: newOutput,
              };
            });
          } else if (data.type === 'log') {
            updateConfigExecutionState(config.id, (prevState) => {
              // Check if this event is for the current generation
              if (prevState.generationId !== generationId) {
                return prevState;
              }
              
              // Check if this EventSource is still the active one
              if (prevState.eventSource !== eventSource) {
                return prevState;
              }
              
              // Ensure we are still running
              if (!prevState.isRunning || prevState.status === 'cancelled') {
                // Don't log if cancelled
                return prevState;
              }
              
              const newOutput = (prevState.rawOutput || '') + data.message + '\n';
              return {
                ...prevState,
                rawOutput: newOutput,
              };
            });
          } else if (data.type === 'complete') {
             updateConfigExecutionState(config.id, (prevState) => {
               // Check if this event is for the current generation
               if (prevState.generationId !== generationId) {
                 return prevState;
               }
               
               // Don't complete if cancelled
               if (!prevState.isRunning || prevState.status === 'cancelled') {
                 return prevState;
               }
               
               const newOutput = (prevState.rawOutput || '') + `\n✅ Generation completed! ${data.num_samples} samples, ${data.num_columns} columns\n`;
               try { eventSource.close(); } catch(e) {}
               
               // Update run record with output file AND llm_requests from backend
               const endTime = new Date().toISOString();
               const duration = (new Date(endTime) - new Date(startTime)) / 1000;
               console.log('Generation complete. LLM requests from backend:', data.llm_requests);
               runsAPI.update(runId, {
                 status: 'completed',
                 end_time: endTime,
                 duration_seconds: duration,
                 output_samples: data.num_samples,
                 output_columns: data.num_columns,
                 output_file: data.output_file ? `outputs/${data.output_file}` : null,
                 llm_requests: data.llm_requests || 0,  // Save LLM requests from backend
               }).catch(err => console.error('Failed to update run record:', err));
               
              return {
                ...prevState,
                status: 'completed', // Explicitly set status
                isRunning: false,
                rawOutput: newOutput,
                outputSamples: data.num_samples,
                duration: `${duration.toFixed(1)}s`,
                result: data,
                eventSource: null,
                completedAt: Date.now(), // Track completion time
              };
             });
           } else if (data.type === 'error') {
            updateConfigExecutionState(config.id, (prevState) => {
              // Check if this event is for the current generation
              if (prevState.generationId !== generationId) {
                return prevState;
              }
              
              // Don't error if cancelled (unless it's the cancellation error itself)
              if ((!prevState.isRunning || prevState.status === 'cancelled') && !data.message.includes('cancelled')) {
                return prevState;
              }
              
              const newOutput = (prevState.rawOutput || '') + `\n❌ Error: ${data.message}\n`;
              try { eventSource.close(); } catch(e) {}
              
              // Update run record
              const endTime = new Date().toISOString();
              const duration = (new Date(endTime) - new Date(startTime)) / 1000;
              runsAPI.update(runId, {
                status: 'failed',
                end_time: endTime,
                duration_seconds: duration,
                error_message: data.message,
               }).catch(err => console.error('Failed to update run record:', err));
               
              return {
                ...prevState,
                status: 'failed', // Explicitly set status
                error: data.message,
                isRunning: false,
                rawOutput: newOutput,
                eventSource: null,
                completedAt: Date.now(), // Track failure time
              };
             });
           }
        } catch (err) {
          console.error('Error parsing event:', err);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        
        updateConfigExecutionState(config.id, (prevState) => {
          // Check if this error is for the current generation
          if (prevState.generationId !== generationId) {
            return prevState;
          }
          
          // If already cancelled, don't update to failed
          if (prevState.status === 'cancelled') {
            return prevState;
          }
          
          // Update run record
          const endTime = new Date().toISOString();
          const duration = (new Date(endTime) - new Date(startTime)) / 1000;
          runsAPI.update(runId, {
            status: 'failed',
            end_time: endTime,
            duration_seconds: duration,
            error_message: 'Connection to server lost',
          }).catch(err => console.error('Failed to update run record:', err));
          
          return {
            ...prevState,
            status: 'failed', // Explicitly set status
            error: 'Connection to server lost',
            isRunning: false,
            rawOutput: (prevState.rawOutput || '') + '\n❌ Connection to server lost\n',
            eventSource: null,
          };
        });
        
        try { eventSource.close(); } catch(e) {}
      };

    } catch (error) {
      console.error('Generation error:', error);
      if (eventSource) {
        eventSource.close();
      }
      
      // Update run record
      const endTime = new Date().toISOString();
      const duration = (new Date(endTime) - new Date(startTime)) / 1000;
      runsAPI.update(runId, {
        status: 'failed',
        end_time: endTime,
        duration_seconds: duration,
        error_message: error.response?.data?.detail || error.message,
      }).catch(err => console.error('Failed to update run record:', err));
      
      updateConfigExecutionState(config.id, (prevState) => ({
        ...(prevState || {}),
        configId: config.id,
        type: 'generate',
        status: 'failed', // Explicitly set status
        error: error.response?.data?.detail || error.message,
        isRunning: false,
        rawOutput: (prevState?.rawOutput || '') + `\n❌ Error: ${error.response?.data?.detail || error.message}\n`,
        result: null,
        eventSource: null,
        runId: runId,
      }));
    }
  };

  /**
   * Handle generate execution (from individual button)
   */
  const handleGenerate = async (config) => {
    let eventSource = null;
    
    try {
      // Open detail view first
      setExpandedConfig(config);
      
      // Start execution
      setExecutionState({
        configId: config.id,
        type: 'generate',
        isRunning: true,
        logs: [],
        rawOutput: '🔧 Loading configuration...\n',
        result: null,
        eventSource: null,
      });

      // Step 1: Load configuration
      await savedConfigAPI.load(config.id);
      
      setExecutionState(prev => ({
        ...prev,
        rawOutput: prev.rawOutput + '✅ Configuration loaded\n\n',
      }));

      // Step 2: Load dataset
      const datasetConfig = config.dataset_configuration || config.dataset_config;
      if (datasetConfig && datasetConfig.data_files && datasetConfig.data_files !== '.') {
        setExecutionState(prev => ({
          ...prev,
          rawOutput: prev.rawOutput + '📊 Loading dataset...\n',
        }));
        
        try {
          await api.post('/api/dataset/load', datasetConfig);
          
          setExecutionState(prev => ({
            ...prev,
            rawOutput: prev.rawOutput + '✅ Dataset loaded\n\n',
          }));
        } catch (datasetError) {
          setExecutionState(prev => ({
            ...prev,
            rawOutput: prev.rawOutput + `⚠️ Dataset loading failed: ${datasetError.response?.data?.detail || datasetError.message}\nContinuing anyway...\n\n`,
          }));
        }
      } else {
        setExecutionState(prev => ({
          ...prev,
          rawOutput: prev.rawOutput + '⚠️ No valid dataset configured, skipping dataset loading\n\n',
        }));
      }

      // Step 3: Start generation
      setExecutionState(prev => ({
        ...prev,
        rawOutput: prev.rawOutput + '🚀 Starting generation...\n\n',
      }));

      // Start generation with streaming (checkpointing enabled by default)
      const maxConcurrency = 100;
      const modelConfigForSaveFreq = config.model_configuration || config.model_config || {};
      const saveFreq = modelConfigForSaveFreq?.additional_params?.save_freq || 10;
      const url = `${STREAM_BASE_URL}/api/flow/generate-stream?config_id=${encodeURIComponent(config.id)}&max_concurrency=${maxConcurrency}&enable_checkpoints=true&save_freq=${saveFreq}&resume_from_checkpoint=false`;
      
      eventSource = new EventSource(url);
      
      setExecutionState(prev => ({
        ...prev,
        eventSource,
      }));
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          setExecutionState(prev => {
            let newOutput = prev.rawOutput;
            
            if (data.type === 'start') {
              newOutput += data.message + '\n';
            } else if (data.type === 'log') {
              // Append log with ANSI codes preserved
              newOutput += data.message + '\n';
            } else if (data.type === 'complete') {
              newOutput += `\n✅ Generation completed! ${data.num_samples} samples, ${data.num_columns} columns\n`;
              eventSource.close();
              return {
                ...prev,
                isRunning: false,
                rawOutput: newOutput,
                result: data,
                eventSource: null,
              };
            } else if (data.type === 'error') {
              newOutput += `\n❌ Error: ${data.message}\n`;
              eventSource.close();
              return {
                ...prev,
                isRunning: false,
                rawOutput: newOutput,
                eventSource: null,
              };
            }
            
            return {
              ...prev,
              rawOutput: newOutput,
            };
          });
        } catch (err) {
          console.error('Error parsing event:', err);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        setExecutionState(prev => ({
          ...prev,
          isRunning: false,
          rawOutput: prev.rawOutput + '\n❌ Connection to server lost\n',
          eventSource: null,
        }));
        eventSource.close();
      };

    } catch (error) {
      console.error('Generation error:', error);
      if (eventSource) {
        eventSource.close();
      }
      setExecutionState(prev => ({
        ...prev,
        isRunning: false,
        rawOutput: prev.rawOutput + `\n❌ Error: ${error.response?.data?.detail || error.message}\n`,
        eventSource: null,
      }));
    }
  };

  /**
   * Get dataset name from config
   */
  const getDatasetName = (config) => {
    const datasetConfig = config.dataset_configuration || config.dataset_config;
    if (datasetConfig?.data_files) {
      const path = datasetConfig.data_files;
      // Extract filename from path
      const parts = path.split('/');
      return parts[parts.length - 1];
    }
    return 'Not specified';
  };

  /**
   * Get model name from config
   */
  const getModelName = (config) => {
    const modelConfig = config.model_configuration || config.model_config;
    if (modelConfig?.model) {
      return modelConfig.model;
    }
    return 'Not configured';
  };

  // Helper functions moved to top


  // If detail view is shown, render it instead of list/table
  if (expandedConfig) {
    const configExecutionState = getExecutionStateForConfig(expandedConfig.id);
    
    return (
      <ConfigurationDetailView
        configuration={expandedConfig}
        onClose={() => {
          // DON'T close EventSource - let execution continue in background
          // Just close the detail view
          setExpandedConfig(null);
        }}
        onRefresh={onRefresh}
        executionState={configExecutionState}
        onDryRun={() => handleDryRun(expandedConfig)}
        onGenerate={() => handleGenerateForConfig(expandedConfig, false)}
        onGenerateFromCheckpoint={() => handleGenerateForConfig(expandedConfig, true)}
        onStop={() => handleStop(expandedConfig)}
        onClearTerminal={handleClearTerminal}
        checkpointInfo={checkpointInfo}
      />
    );
  }

  return (
    <div>
      {/* Show toolbar when not in detail view */}
      {renderToolbar && renderToolbar()}
      
       <ConfigurationTable
         configurations={configurations}
         selectedConfigs={selectedConfigs}
         onToggleSelection={toggleSelection}
         onToggleSelectAll={() => {
           if (selectedConfigs.length === configurations.length) {
             setSelectedConfigs([]);
             if (onSelectedConfigsChange) onSelectedConfigsChange([]);
           } else {
             const allIds = configurations.map(c => c.id);
             setSelectedConfigs(allIds);
             if (onSelectedConfigsChange) onSelectedConfigsChange(allIds);
           }
         }}
         onDryRun={handleDryRun}
         onGenerate={handleGenerateForConfig}
         onEdit={onEdit}
         onDelete={handleDelete}
         onStop={handleStop}
         onClone={onClone}
         isRunning={isConfigRunning}
         onFlowNameClick={handleFlowNameClick}
         executionStates={executionStates}
       />
    </div>
  );
};

export default ConfigurationList;
