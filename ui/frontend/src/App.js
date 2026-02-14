import React, { useState, useEffect, useRef } from 'react';
import {
  PageSidebar,
  PageSidebarBody,
  Nav,
  NavList,
  NavItem,
  Alert,
  AlertVariant,
  Spinner,
} from '@patternfly/react-core';
import AppHeader from './components/AppHeader';
import HomeDashboard from './components/HomeDashboard';
import Dashboard from './components/Dashboard';
import DataGenerationFlowsPage from './components/DataGenerationFlowsPage';
import FlowRunsHistoryPage from './components/FlowRunsHistoryPage';
import UnifiedFlowWizard from './components/UnifiedFlowWizard';
import { healthCheck, executionAPI, runsAPI, API_BASE_URL } from './services/api';
import { useNotifications } from './contexts/NotificationContext';

const App = () => {
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const [healthError, setHealthError] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Persist activeItem to sessionStorage so navigation state survives refresh
  const [activeItem, setActiveItem] = useState(() => {
    try {
      const savedActiveItem = sessionStorage.getItem('app_active_item');
      if (savedActiveItem) {
        return savedActiveItem;
      }
    } catch (error) {
      console.error('Failed to load active item from sessionStorage:', error);
    }
    return 'home';  // Changed from 'flows' to 'home'
  });
  
  // Global execution states (persists across all pages and page refreshes)
  const [executionStates, setExecutionStates] = useState(() => {
    // Load persisted execution states from localStorage on mount
    // NOTE: We don't mark running flows as cancelled here - we'll check with backend first
    try {
      const saved = localStorage.getItem('sdg_execution_states');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Keep running states as-is for now - we'll verify with backend in useEffect
        // Just mark them as "pending_reconnection" so UI knows to check
        Object.keys(parsed).forEach(configId => {
          if (parsed[configId].isRunning) {
            parsed[configId] = {
              ...parsed[configId],
              pendingReconnection: true, // Flag to check with backend
            };
          }
        });
        return parsed;
      }
    } catch (error) {
      console.error('Failed to load execution states from localStorage:', error);
    }
    return {};
  });
  
  // Track if we've checked for running generations after mount
  const [hasCheckedRunningGenerations, setHasCheckedRunningGenerations] = useState(false);
  
  // Wizard state (for resuming and navigation)
  const [wizardData, setWizardData] = useState(null);
  const [editingConfig, setEditingConfig] = useState(null);
  
  // State for triggering auto-run after wizard completion (Save and Run)
  const [autoRunConfig, setAutoRunConfig] = useState(null);
  
  // Notification system
  const { addSuccessNotification, addErrorNotification, addWarningNotification } = useNotifications();
  
  // Track previous execution states to detect changes
  const previousExecutionStatesRef = useRef({});
  
  // Ref to read executionStates without triggering re-runs in effects
  const executionStatesRef = useRef(executionStates);
  useEffect(() => { executionStatesRef.current = executionStates; }, [executionStates]);
  
  // Track active EventSource instances for cleanup on unmount
  const activeEventSourcesRef = useRef(new Map());

  /**
   * Update execution state for a specific config
   */
  const updateExecutionState = (configId, updates) => {
    setExecutionStates(prev => {
      const newState = {
        ...prev,
        [configId]: typeof updates === 'function' 
          ? updates(prev[configId] || {})
          : {
              ...(prev[configId] || {}),
              ...updates
            }
      };
      return newState;
    });
  };

  /**
   * Get execution state for a config
   */
  const getExecutionState = (configId) => {
    return executionStates[configId] || null;
  };
  
  /**
   * Check backend health on mount and periodically
   */
  useEffect(() => {
    let timeoutId;
    
    const checkHealth = async () => {
      try {
        await healthCheck();
        setIsBackendHealthy(true);
        setHealthError(null);
        // Schedule next check in 30s if healthy
        timeoutId = setTimeout(checkHealth, 30000);
      } catch (error) {
        setIsBackendHealthy(false);
        setHealthError(error.message);
        // Schedule next check in 2s if unhealthy (fast recovery)
        timeoutId = setTimeout(checkHealth, 2000);
      } finally {
        setIsCheckingHealth(false);
      }
    };
    
    // Initial health check
    checkHealth();
    
    return () => clearTimeout(timeoutId);
  }, []);
  
  /**
   * Persist activeItem to sessionStorage whenever it changes
   * This ensures the navigation state survives page refresh
   */
  useEffect(() => {
    try {
      sessionStorage.setItem('app_active_item', activeItem);
    } catch (error) {
      console.error('Failed to save active item to sessionStorage:', error);
    }
  }, [activeItem]);

  /**
   * Persist execution states to localStorage whenever they change
   * This ensures flow status (completed, failed, etc.) survives page refresh
   */
  useEffect(() => {
    try {
      // Create a serializable copy with only essential fields
      const serializableStates = {};
      Object.keys(executionStates).forEach(configId => {
        const state = executionStates[configId];
        serializableStates[configId] = {
          configId: state.configId,
          configName: state.configName,
          flowName: state.flowName,
          type: state.type,
          status: state.status,
          isRunning: state.isRunning,
          runId: state.runId,
          startTime: state.startTime,
          completedAt: state.completedAt,
          cancelledAt: state.cancelledAt,
          error: state.error,
          duration: state.duration,
          outputSamples: state.outputSamples,
          // Persist result object for download functionality
          result: state.result,
          // Limit rawOutput to last 5000 chars to avoid localStorage limits
          rawOutput: state.rawOutput ? state.rawOutput.slice(-5000) : '',
          // Don't persist EventSource objects (non-serializable)
          eventSource: null,
          // Don't persist pendingReconnection flag
          pendingReconnection: undefined,
        };
      });
      localStorage.setItem('sdg_execution_states', JSON.stringify(serializableStates));
    } catch (error) {
      console.error('Failed to save execution states to localStorage:', error);
    }
  }, [executionStates]);

  /**
   * Check for running generations after backend is healthy and reconnect
   * This allows flows to continue after page refresh
   */
  useEffect(() => {
    if (!isBackendHealthy || hasCheckedRunningGenerations) {
      return;
    }

    const checkAndReconnect = async () => {
      try {
        console.log('🔄 Checking for running generations after page refresh...');
        
        // Get list of running generations from backend
        const statusResponse = await executionAPI.getGenerationStatus();
        const runningGenerations = statusResponse.running_generations || [];
        
        console.log(`Found ${runningGenerations.length} running generation(s)`);
        
        // For each running generation, try to reconnect
        for (const gen of runningGenerations) {
          const configId = gen.config_id;
          const currentState = executionStatesRef.current[configId];
          
          if (currentState?.pendingReconnection || currentState?.isRunning) {
            console.log(`🔗 Reconnecting to generation for config: ${configId}`);
            
            // Create new EventSource for reconnection
            const reconnectUrl = `${API_BASE_URL}/api/flow/reconnect-stream?config_id=${encodeURIComponent(configId)}`;
            const eventSource = new EventSource(reconnectUrl);
            
            // Track EventSource for cleanup on unmount
            activeEventSourcesRef.current.set(configId, eventSource);
            
            // Helper to cleanup EventSource
            const cleanupEventSource = () => {
              eventSource.close();
              activeEventSourcesRef.current.delete(configId);
            };
            
            // Update state to show as running and connected
            setExecutionStates(prev => ({
              ...prev,
              [configId]: {
                ...prev[configId],
                isRunning: true,
                status: 'running',
                pendingReconnection: false,
                eventSource: eventSource,
                rawOutput: (prev[configId]?.rawOutput || '') + '\n🔄 Reconnected after page refresh...\n',
              }
            }));
            
            // Handle messages
            eventSource.onmessage = (event) => {
              try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'reconnected') {
                  setExecutionStates(prev => ({
                    ...prev,
                    [configId]: {
                      ...prev[configId],
                      rawOutput: (prev[configId]?.rawOutput || '') + data.message + '\n',
                    }
                  }));
                } else if (data.type === 'log') {
                  setExecutionStates(prev => ({
                    ...prev,
                    [configId]: {
                      ...prev[configId],
                      rawOutput: (prev[configId]?.rawOutput || '') + data.message + '\n',
                    }
                  }));
                } else if (data.type === 'complete') {
                  cleanupEventSource();
                  
                  // Update run record with llm_requests from backend
                  const runId = currentState?.runId;
                  if (runId) {
                    const endTime = new Date().toISOString();
                    const startTime = currentState?.startTime;
                    const duration = startTime ? (new Date(endTime) - new Date(startTime)) / 1000 : null;
                    
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
                  }
                  
                  setExecutionStates(prev => ({
                    ...prev,
                    [configId]: {
                      ...prev[configId],
                      isRunning: false,
                      status: 'completed',
                      rawOutput: (prev[configId]?.rawOutput || '') + `\n✅ Generation completed! ${data.num_samples} samples, ${data.num_columns} columns\n`,
                      result: data,
                      outputSamples: data.num_samples,
                      eventSource: null,
                      completedAt: Date.now(),
                    }
                  }));
                } else if (data.type === 'error') {
                  cleanupEventSource();
                  
                  // Update run record
                  const runId = currentState?.runId;
                  if (runId) {
                    runsAPI.update(runId, {
                      status: 'failed',
                      error_message: data.message,
                    }).catch(err => console.error('Failed to update run record:', err));
                  }
                  
                  setExecutionStates(prev => ({
                    ...prev,
                    [configId]: {
                      ...prev[configId],
                      isRunning: false,
                      status: 'failed',
                      error: data.message,
                      rawOutput: (prev[configId]?.rawOutput || '') + `\n❌ Error: ${data.message}\n`,
                      eventSource: null,
                    }
                  }));
                }
              } catch (err) {
                console.error('Error parsing reconnect event:', err);
              }
            };
            
            eventSource.onerror = (error) => {
              console.error('Reconnect EventSource error:', error);
              cleanupEventSource();
              
              setExecutionStates(prev => ({
                ...prev,
                [configId]: {
                  ...prev[configId],
                  isRunning: false,
                  status: 'failed',
                  error: 'Connection lost during reconnection',
                  rawOutput: (prev[configId]?.rawOutput || '') + '\n❌ Connection lost\n',
                  eventSource: null,
                }
              }));
            };
          }
        }
        
        // For any states that were marked as running but backend says they're not running,
        // check if they completed or failed
        Object.keys(executionStatesRef.current).forEach(configId => {
          const state = executionStatesRef.current[configId];
          if (state?.pendingReconnection && !runningGenerations.find(g => g.config_id === configId)) {
            // This generation is no longer running on backend
            // Check runs history to see what happened
            console.log(`⚠️ Generation for config ${configId} is no longer running on backend`);
            
            setExecutionStates(prev => ({
              ...prev,
              [configId]: {
                ...prev[configId],
                isRunning: false,
                status: prev[configId]?.status === 'running' ? 'cancelled' : prev[configId]?.status,
                pendingReconnection: false,
                rawOutput: (prev[configId]?.rawOutput || '') + '\n⚠️ Generation ended while disconnected\n',
                eventSource: null,
              }
            }));
          }
        });
        
      } catch (error) {
        console.error('Error checking for running generations:', error);
        
        // Mark all pending reconnections as cancelled since we couldn't check
        setExecutionStates(prev => {
          const updated = { ...prev };
          Object.keys(updated).forEach(configId => {
            if (updated[configId]?.pendingReconnection) {
              updated[configId] = {
                ...updated[configId],
                isRunning: false,
                status: 'cancelled',
                pendingReconnection: false,
                rawOutput: (updated[configId]?.rawOutput || '') + '\n⚠️ Could not reconnect after refresh\n',
              };
            }
          });
          return updated;
        });
      }
      
      setHasCheckedRunningGenerations(true);
    };
    
    checkAndReconnect();
  }, [isBackendHealthy, hasCheckedRunningGenerations]); // executionStates accessed via ref

  /**
   * Cleanup all active EventSources on component unmount
   */
  useEffect(() => {
    return () => {
      // Close all tracked EventSources to prevent memory leaks
      activeEventSourcesRef.current.forEach((eventSource, configId) => {
        console.log(`🧹 Cleaning up EventSource for config: ${configId}`);
        eventSource.close();
      });
      activeEventSourcesRef.current.clear();
    };
  }, []);

  /**
   * Monitor execution state changes and trigger notifications
   */
  useEffect(() => {
    const previousStates = previousExecutionStatesRef.current;
    
    // Check each execution state for status changes
    Object.keys(executionStates).forEach(configId => {
      const currentState = executionStates[configId];
      const previousState = previousStates[configId];
      
      // Skip if no previous state (first time seeing this config)
      if (!previousState) {
        // console.log('Skipping notification (first state):', configId);
        return;
      }
      
      // Skip notifications if this is a restart (user starting a new generation)
      // Check this FIRST before any notification logic
      if (currentState.isRestarting) {
        // Clear the restart flag after a short delay
        setTimeout(() => {
          setExecutionStates(prev => ({
            ...prev,
            [configId]: {
              ...prev[configId],
              isRestarting: false
            }
          }));
        }, 1000); // Give enough time to skip all notification checks
        return;
      }
      
      // Check for status transitions
      const prevStatus = previousState.status;
      const currStatus = currentState.status;
      
      // Detect completion
      if ((prevStatus === 'running' || prevStatus === 'generating') && currStatus === 'completed') {
        const flowName = currentState.flowName || currentState.configName || 'Flow';
        const duration = currentState.duration || 'Unknown';
        const outputSamples = currentState.outputSamples || 'N/A';
        
        addSuccessNotification(
          `✅ ${flowName} completed!`,
          `Generated ${outputSamples} samples in ${duration}`,
          10000
        );
      }
      
      // Detect failure
      if ((prevStatus === 'running' || prevStatus === 'generating') && currStatus === 'failed') {
        const flowName = currentState.flowName || currentState.configName || 'Flow';
        const error = currentState.error || 'Unknown error';
        
        addErrorNotification(
          `❌ ${flowName} failed`,
          `Error: ${error.substring(0, 100)}${error.length > 100 ? '...' : ''}`,
          12000
        );
      }
      
      // Skip notifications for flows that have been completed/failed for more than 5 seconds
      // This prevents false notifications when old flows get state updates
      if ((prevStatus === 'completed' || prevStatus === 'failed') && 
          currentState.completedAt && 
          (Date.now() - currentState.completedAt > 5000)) {
        return;
      }
      
      // Detect cancellation - but only if it wasn't user-initiated
      // User-initiated cancellations have cancelledAt timestamp and we don't want to notify for those
      // Also don't notify if we're transitioning from a completed/failed state (not from running)
      if (currStatus === 'cancelled' && prevStatus === 'running') {
        // Check if this was a user-initiated cancellation (has cancelledAt timestamp)
        // If cancelledAt exists and is recent (within last 5 seconds), it's user-initiated
        const wasUserInitiated = currentState.cancelledAt && 
          (Date.now() - currentState.cancelledAt < 5000);
        
        if (!wasUserInitiated) {
          // Only show notification for unexpected cancellations (e.g., backend crash)
          const flowName = currentState.flowName || currentState.configName || 'Flow';
          
          addWarningNotification(
            `⚠️ ${flowName} cancelled unexpectedly`,
            'Flow execution was stopped by the system',
            8000
          );
        }
      }
    });
    
    // Update previous states reference
    previousExecutionStatesRef.current = JSON.parse(JSON.stringify(executionStates));
  }, [executionStates, addSuccessNotification, addErrorNotification, addWarningNotification]);

  
  /**
   * Render loading state
   */
  if (isCheckingHealth) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <Spinner size="xl" />
        <p>Connecting to SDG Hub API...</p>
      </div>
    );
  }
  
  /**
   * Render error state if backend is not healthy
   */
  if (!isBackendHealthy) {
    const isNetworkError = healthError?.includes('Network Error') || healthError?.includes('Failed to fetch');
    
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        padding: '2rem'
      }}>
        <Alert
          variant={isNetworkError ? AlertVariant.warning : AlertVariant.danger}
          title={isNetworkError ? "Reconnecting to server..." : "Cannot connect to backend API"}
          isInline
        >
          <p>
            {isNetworkError 
              ? "The server is restarting or temporarily unavailable. Auto-reconnecting..." 
              : "The SDG Hub API server is not responding."}
          </p>
          <div style={{ marginTop: '1rem', fontSize: '0.9em', opacity: 0.8 }}>
            <p><strong>Status:</strong> {healthError || 'Unknown error'}</p>
            <p style={{ marginTop: '0.5rem' }}>If this persists, run: <code>cd backend && python api_server.py</code></p>
          </div>
        </Alert>
      </div>
    );
  }
  /**
   * Handle wizard completion
   */
  const handleWizardComplete = (newConfig, options = {}) => {
    // Navigate back to flows page
    setActiveItem('flows');
    setWizardData(null);
    setEditingConfig(null);
    
    // If shouldRun is true, set up auto-run for the saved config
    if (options.shouldRun && newConfig) {
      setAutoRunConfig(newConfig);
    }
    // Flows page will auto-refresh and show the new configuration
  };
  
  /**
   * Handle wizard cancel
   */
  const handleWizardCancel = () => {
    setActiveItem('flows');
    setWizardData(null);
    setEditingConfig(null);
  };
  
  /**
   * Render main application content based on active page
   */
  const renderPageContent = () => {
    switch (activeItem) {
      case 'home':
        return (
          <HomeDashboard
            onNavigate={(page, data) => {
              setActiveItem(page);
              if (page === 'configure-flow') {
                setWizardData(data);
                setEditingConfig(null);
              }
            }}
          />
        );
      case 'dashboard':
        return (
          <Dashboard
            executionStates={executionStates}
            onUpdateExecutionState={updateExecutionState}
            onNavigate={(page, data) => {
              setActiveItem(page);
              if (page === 'configure-flow') {
                setWizardData(data);
                setEditingConfig(null);
              }
            }}
          />
        );
      case 'flows':
        return (
          <DataGenerationFlowsPage 
            executionStates={executionStates}
            onUpdateExecutionState={updateExecutionState}
            getExecutionState={getExecutionState}
            onNavigate={(page, data) => {
              setActiveItem(page);
              if (page === 'configure-flow') {
                setWizardData(data);
                setEditingConfig(null); // Clear editing config when creating new flow
              }
            }}
            onEditConfiguration={(config, wizardData) => {
              setEditingConfig(config);
              if (wizardData) {
                setWizardData(wizardData);
              }
              setActiveItem('configure-flow');
            }}
            autoRunConfig={autoRunConfig}
            onAutoRunComplete={() => setAutoRunConfig(null)}
          />
        );
      case 'configure-flow':
        return (
          <UnifiedFlowWizard
            key={`wizard-${editingConfig?.id || wizardData?.draftData?.id || wizardData?.clonedConfig?.id || 'new'}`}
            wizardData={wizardData}
            editingConfig={editingConfig}
            onComplete={handleWizardComplete}
            onCancel={handleWizardCancel}
          />
        );
      case 'flow-runs':
        return (
          <FlowRunsHistoryPage 
            executionStates={executionStates}
            onUpdateExecutionState={updateExecutionState}
            getExecutionState={getExecutionState}
          />
        );
      default:
        return (
          <DataGenerationFlowsPage 
            executionStates={executionStates}
            onUpdateExecutionState={updateExecutionState}
            getExecutionState={getExecutionState}
            onNavigate={(page, data) => {
              setActiveItem(page);
              if (page === 'configure-flow') {
                setWizardData(data);
                setEditingConfig(null); // Clear editing config when creating new flow
              }
            }}
            autoRunConfig={autoRunConfig}
            onAutoRunComplete={() => setAutoRunConfig(null)}
          />
        );
    }
  };
  
  /**
   * Render main application
   */
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <AppHeader 
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
      />
      
      {/* Main Content Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        {isSidebarOpen && (
          <div style={{
            width: '250px',
            backgroundColor: '#212427',
            color: 'white',
            flexShrink: 0,
            overflowY: 'auto'
          }}>
            <Nav aria-label="Main navigation" theme="dark">
              <NavList>
                <NavItem
                  itemId="home"
                  isActive={activeItem === 'home'}
                  onClick={() => setActiveItem('home')}
                >
                  Home
                </NavItem>
                <NavItem
                  itemId="dashboard"
                  isActive={activeItem === 'dashboard'}
                  onClick={() => setActiveItem('dashboard')}
                >
                  Dashboard
                </NavItem>
                <NavItem
                  itemId="flows"
                  isActive={activeItem === 'flows' || activeItem === 'configure-flow'}
                  onClick={() => {
                    // Check if user is currently in the wizard
                    if (activeItem === 'configure-flow') {
                      const hasWizardSession = sessionStorage.getItem('wizard_session_state');
                      if (hasWizardSession) {
                        try {
                          const session = JSON.parse(hasWizardSession);
                          const hasContent = session.sourceType || session.selectedFlow || session.modelConfig?.model || session.datasetConfig?.data_files;
                          
                          if (hasContent) {
                            // Show confirmation dialog
                            const choice = window.confirm(
                              'You have unsaved progress in the wizard.\n\n' +
                              'Click "OK" to save your progress and return to flows list.\n' +
                              'Click "Cancel" to stay in the wizard.'
                            );
                            
                            if (choice) {
                              // User wants to save and go to flows - the wizard will auto-save on close
                              // Clear wizard session and go to flows
                              setActiveItem('flows');
                            }
                            // If cancelled, stay in wizard (do nothing)
                            return;
                          }
                        } catch (e) {
                          console.error('Error parsing wizard session:', e);
                        }
                      }
                      // No meaningful content, just go to flows
                      sessionStorage.removeItem('wizard_session_state');
                      setActiveItem('flows');
                    } else {
                      // Not in wizard - check if there's a saved wizard session to restore
                      const hasWizardSession = sessionStorage.getItem('wizard_session_state');
                      if (hasWizardSession) {
                        // Return to wizard
                        setActiveItem('configure-flow');
                      } else {
                        // Return to main flows table
                        setActiveItem('flows');
                      }
                    }
                  }}
                >
                  Data Generation Flows
                </NavItem>
                <NavItem
                  itemId="flow-runs"
                  isActive={activeItem === 'flow-runs'}
                  onClick={() => setActiveItem('flow-runs')}
                >
                  Flow Runs History
                </NavItem>
              </NavList>
            </Nav>
          </div>
        )}
        
        {/* Page Content */}
        <div style={{ flex: 1, overflowY: 'auto', backgroundColor: '#f5f5f5' }}>
          {renderPageContent()}
        </div>
      </div>
    </div>
  );
};

export default App;
