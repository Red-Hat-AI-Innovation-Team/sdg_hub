import React, { useState, useEffect } from 'react';
import {
  Page,
  PageSection,
  Wizard,
  WizardStep,
  WizardFooterWrapper,
  useWizardContext,
  Button,
  Alert,
  AlertVariant,
  Title,
  Card,
  CardBody,
  Radio,
  SearchInput,
  Spinner,
  EmptyState,
  EmptyStateIcon,
  EmptyStateHeader,
  EmptyStateBody,
  List,
  ListItem,
  Badge,
  Modal,
  ModalVariant,
} from '@patternfly/react-core';
import { PlayIcon } from '@patternfly/react-icons';
import {
  CubesIcon,
  PlusCircleIcon,
  CopyIcon,
  EditIcon,
} from '@patternfly/react-icons';
import FlowSelectionStep from './steps/FlowSelectionStep';
import ModelConfigurationStep from './steps/ModelConfigurationStep';
import DatasetConfigurationStep from './steps/DatasetConfigurationStep';
import DryRunSettingsStep from './steps/DryRunSettingsStep';
import ReviewStep from './steps/ReviewStep';
import FlowBuilderPage from './flowCreator/FlowBuilderPage';
import { flowAPI, savedConfigAPI } from '../services/api';

/**
 * Custom Footer for Build Flow step with Save Changes button
 */
const BuildFlowFooter = ({ flowBuilderSaveInfo, selectedFlow }) => {
  const { goToNextStep, goToPrevStep } = useWizardContext();
  
  return (
    <WizardFooterWrapper>
      <Button variant="secondary" onClick={goToPrevStep}>
        Back
      </Button>
      {/* Save Changes/Save Flow button - only show when there are blocks */}
      {flowBuilderSaveInfo?.needsSave && (
        <Button
          variant={flowBuilderSaveInfo?.hasUnsavedChanges || !selectedFlow ? "warning" : "secondary"}
          onClick={() => {
            if (flowBuilderSaveInfo?.isEditMode && flowBuilderSaveInfo?.triggerQuickSave) {
              flowBuilderSaveInfo.triggerQuickSave();
            } else if (flowBuilderSaveInfo?.openMetadataModal) {
              flowBuilderSaveInfo.openMetadataModal();
            }
          }}
          style={flowBuilderSaveInfo?.hasUnsavedChanges || !selectedFlow ? { 
            backgroundColor: '#f0ab00', 
            color: '#151515',
            borderColor: '#f0ab00'
          } : {}}
        >
          {flowBuilderSaveInfo?.isEditMode ? 'Save Changes' : 'Save Flow'}
        </Button>
      )}
      <Button
        variant="primary"
        onClick={goToNextStep}
        isDisabled={!selectedFlow}
      >
        Next
      </Button>
    </WizardFooterWrapper>
  );
};

// Session storage key for wizard state persistence
const WIZARD_SESSION_KEY = 'wizard_session_state';

/**
 * Load saved wizard session state from sessionStorage
 */
const loadWizardSessionState = () => {
  try {
    const saved = sessionStorage.getItem(WIZARD_SESSION_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (error) {
    console.error('Failed to load wizard session state:', error);
    sessionStorage.removeItem(WIZARD_SESSION_KEY);
  }
  return null;
};

/**
 * Clear wizard session state
 */
const clearWizardSessionState = () => {
  sessionStorage.removeItem(WIZARD_SESSION_KEY);
};

/**
 * Unified Flow Wizard
 * 
 * Combines flow creation and configuration in a single wizard experience.
 * Now displayed as a full page instead of a modal.
 * 
 * Steps:
 * 1. Choose Flow Source (use existing / create custom)
 * 2a. Select Existing Flow (if using existing)
 * 2b. Build Custom Flow (if creating custom) - embeds FlowBuilderPage
 * 3. Configure Model
 * 4. Configure Dataset
 * 5. Dry Run Settings (Optional)
 * 6. Review & Save
 */
const UnifiedFlowWizard = ({ wizardData, editingConfig, onComplete, onCancel }) => {
  // Try to restore session state first (for refresh/navigation scenarios)
  const savedSessionState = React.useMemo(() => loadWizardSessionState(), []);
  
  // Check if we should restore session state
  // Restore if we have saved state AND no explicit intent to edit/create
  // (clicking "Configure Flow" button clears session, so this only triggers on refresh/nav back)
  const shouldRestoreSession = savedSessionState && !editingConfig && !wizardData?.sourceType && !wizardData?.resumeDraft && !wizardData?.isCloning;
  
  // Initialize sourceType based on props or restored session
  const initialSourceType = (() => {
    // First priority: restored session state
    if (shouldRestoreSession && savedSessionState.sourceType) {
      return savedSessionState.sourceType;
    }
    if (wizardData?.sourceType) {
      return wizardData.sourceType;
    }
    if (wizardData?.resumeDraft) return 'draft';
    if (editingConfig) {
      // Multiple ways to detect custom flows
      const hasCustomSuffix = editingConfig.flow_name?.includes('(Custom)');
      const hasCustomPath = editingConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = editingConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCustomPath || hasCustomFlag;
      const result = isCustomFlow ? 'clone' : 'existing';
      return result;
    }
    return null;
  })();
  
  // Source selection state
  const [sourceType, setSourceType] = useState(initialSourceType);
  
  // Flow state - restore from session if available
  const [selectedFlow, setSelectedFlow] = useState(shouldRestoreSession ? savedSessionState.selectedFlow : null);
  const [createdFlow, setCreatedFlow] = useState(shouldRestoreSession ? savedSessionState.createdFlow : null);
  const [clonedFlow, setClonedFlow] = useState(shouldRestoreSession ? savedSessionState.clonedFlow : null);
  const [draftFlow, setDraftFlow] = useState(shouldRestoreSession ? savedSessionState.draftFlow : null);
  
  // Configuration state - restore from session if available
  const [modelConfig, setModelConfig] = useState(
    shouldRestoreSession && savedSessionState.modelConfig ? savedSessionState.modelConfig : {}
  );
  const [datasetConfig, setDatasetConfig] = useState(
    shouldRestoreSession && savedSessionState.datasetConfig ? savedSessionState.datasetConfig : {}
  );
  const [dryRunConfig, setDryRunConfig] = useState(
    shouldRestoreSession && savedSessionState.dryRunConfig ? savedSessionState.dryRunConfig : { 
      sample_size: 2, 
      enable_time_estimation: true, 
      max_concurrency: 10 
    }
  );

  // Ref to trigger save from FlowBuilderPage
  const flowBuilderSaveRef = React.useRef(null);
  const [flowBuilderSaveInfo, setFlowBuilderSaveInfo] = useState(null);
  
  // Track current step for session persistence
  const [currentStepId, setCurrentStepId] = useState(
    shouldRestoreSession && savedSessionState.currentStepId ? savedSessionState.currentStepId : null
  );
  
  // Calculate initial step ID immediately based on props or restored session
  const calculateInitialStepId = () => {
    // First priority: restored session state
    if (shouldRestoreSession && savedSessionState.currentStepId) {
      return savedSessionState.currentStepId;
    }
    if (wizardData?.startStepName) {
      return wizardData.startStepName;
    }
    if (wizardData?.resumeDraft) {
      return 'build-flow';
    }
    if (editingConfig) {
      // Multiple ways to detect custom flows
      const hasCustomSuffix = editingConfig.flow_name?.includes('(Custom)');
      const hasCustomPath = editingConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = editingConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCustomPath || hasCustomFlag;
      
      if (isCustomFlow) {
        return 'build-flow';
      }
      
      // For existing flows, go to select-existing (as requested by user)
      return 'select-existing';
    }
    return 'source-selection'; // Fresh start
  };
  
  // UI state
  const [errorMessage, setErrorMessage] = useState(null);
  const [isWizardOpen, setIsWizardOpen] = useState(true);
  const [initialStepId, setInitialStepId] = useState(calculateInitialStepId());
  
  // For cloning - restore from session if available
  const [availableFlows, setAvailableFlows] = useState([]);
  const [flowsLoading, setFlowsLoading] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const [selectedCloneFlow, setSelectedCloneFlow] = useState(
    shouldRestoreSession ? savedSessionState.selectedCloneFlow : null
  );
  const [showCloneModal, setShowCloneModal] = useState(false);
  
  // For draft selection
  const [availableDrafts, setAvailableDrafts] = useState([]);
  const [showDraftModal, setShowDraftModal] = useState(false);
  const [selectedDraftId, setSelectedDraftId] = useState(
    shouldRestoreSession ? savedSessionState.selectedDraftId : null
  );
  
  // Validation state - restore from session if available
  const [stepValidation, setStepValidation] = useState(
    shouldRestoreSession && savedSessionState.stepValidation ? savedSessionState.stepValidation : {
      0: false, // Source selection
      1: false, // Flow selection/creation
      2: false, // Model configuration
      3: false, // Dataset configuration
      4: true,  // Dry run (optional)
      5: true,  // Review
    }
  );
  
  // Track initial config values to detect changes (for "Save and Exit" button)
  const [initialConfigSnapshot, setInitialConfigSnapshot] = useState(null);
  
  // Check if we're in edit mode (editing an existing configuration)
  const isEditMode = !!editingConfig;
  
  /**
   * Save wizard state to sessionStorage whenever important state changes
   */
  useEffect(() => {
    // Don't save if wizard is closed or we're in the middle of initialization
    if (!isWizardOpen) return;
    
    // Only save if there's meaningful state to save
    const hasState = sourceType || selectedFlow || modelConfig?.model || datasetConfig?.data_files;
    if (!hasState) return;
    
    const stateToSave = {
      sourceType,
      selectedFlow,
      createdFlow,
      clonedFlow,
      draftFlow,
      modelConfig,
      datasetConfig,
      dryRunConfig,
      stepValidation,
      currentStepId,
      selectedCloneFlow,
      selectedDraftId,
      // Save context about original editing config (if any)
      editingConfigId: editingConfig?.id || null,
      editingConfigName: editingConfig?.flow_name || selectedFlow?.name || null,
      timestamp: Date.now(),
    };
    
    try {
      sessionStorage.setItem(WIZARD_SESSION_KEY, JSON.stringify(stateToSave));
    } catch (error) {
      console.error('Failed to save wizard session state:', error);
    }
  }, [
    isWizardOpen,
    sourceType,
    selectedFlow,
    createdFlow,
    clonedFlow,
    draftFlow,
    modelConfig,
    datasetConfig,
    dryRunConfig,
    stepValidation,
    currentStepId,
    selectedCloneFlow,
    selectedDraftId,
    editingConfig,
  ]);
  
  /**
   * Capture initial config snapshot when editing (for change detection)
   */
  useEffect(() => {
    if (isEditMode && editingConfig && !initialConfigSnapshot) {
      // Capture the initial state when editing starts
      const snapshot = {
        modelConfig: editingConfig.model_configuration || editingConfig.model_config || {},
        datasetConfig: editingConfig.dataset_configuration || editingConfig.dataset_config || {},
        dryRunConfig: editingConfig.dry_run_configuration || { sample_size: 2, enable_time_estimation: true, max_concurrency: 10 },
      };
      setInitialConfigSnapshot(snapshot);
    }
  }, [isEditMode, editingConfig, initialConfigSnapshot]);
  
  /**
   * Check if any changes were made compared to initial config
   */
  const hasChanges = React.useMemo(() => {
    if (!isEditMode || !initialConfigSnapshot) return false;
    
    // Compare model config
    const modelChanged = JSON.stringify(modelConfig) !== JSON.stringify(initialConfigSnapshot.modelConfig);
    
    // Compare dataset config
    const datasetChanged = JSON.stringify(datasetConfig) !== JSON.stringify(initialConfigSnapshot.datasetConfig);
    
    // Compare dry run config
    const dryRunChanged = JSON.stringify(dryRunConfig) !== JSON.stringify(initialConfigSnapshot.dryRunConfig);
    
    return modelChanged || datasetChanged || dryRunChanged;
  }, [isEditMode, initialConfigSnapshot, modelConfig, datasetConfig, dryRunConfig]);

  /**
   * Load all saved drafts from localStorage
   */
  const loadAllDrafts = () => {
    try {
      const draftsJson = localStorage.getItem('wizard_drafts');
      if (draftsJson) {
        const drafts = JSON.parse(draftsJson);
        return Array.isArray(drafts) ? drafts : [];
      }
    } catch (error) {
      console.error('Failed to parse drafts:', error);
      localStorage.removeItem('wizard_drafts');
    }
    return [];
  };
  
  /**
   * Load drafts and pre-populate wizard data on mount
   */
  useEffect(() => {
    // Load all available drafts
    const drafts = loadAllDrafts();
    setAvailableDrafts(drafts);
    
    // If editing an existing configuration
    if (editingConfig) {
      
      // Multiple ways to detect custom flows
      const hasCustomSuffix = editingConfig.flow_name?.includes('(Custom)');
      const hasCustomPath = editingConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = editingConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCustomPath || hasCustomFlag;
      
      
      const modelConfig = editingConfig.model_configuration || editingConfig.model_config || {};
      const datasetConfig = editingConfig.dataset_configuration || editingConfig.dataset_config || {};
      
      const hasModel = modelConfig.model && 
                       modelConfig.model !== 'Not configured' && 
                       modelConfig.model !== 'Not specified';
      const hasDataset = datasetConfig.data_files && 
                         datasetConfig.data_files !== 'Not configured' && 
                         datasetConfig.data_files !== 'Not specified';
      
      // Pre-populate wizard with existing data
      setSelectedFlow({
        name: editingConfig.flow_name,
        id: editingConfig.flow_id || editingConfig.id,
        path: editingConfig.flow_path,
        tags: editingConfig.tags || [],
        isCustomFlow: isCustomFlow,
      });
      
      if (hasModel) {
        setModelConfig(modelConfig);
      }
      
      if (hasDataset) {
        setDatasetConfig(datasetConfig);
      }
      
      // For custom flows, load the flow blocks for editing
      if (isCustomFlow) {
        loadCustomFlowForEditing(editingConfig.flow_name);
      }
      
      // Mark steps as valid based on what's configured
      markStepValid(0, true); // Source selected
      markStepValid(1, isCustomFlow ? false : true); // For custom flows, user may want to modify in Build Flow
      markStepValid(2, hasModel);
      markStepValid(3, hasDataset);
      
      return;
    }
    
    // If navigated to clone a configuration
    if (wizardData?.isCloning && wizardData?.clonedConfig) {
      const clonedConfig = wizardData.clonedConfig;
      const modelConfigData = clonedConfig.model_configuration || clonedConfig.model_config || {};
      const datasetConfigData = clonedConfig.dataset_configuration || clonedConfig.dataset_config || {};
      
      // Multiple ways to detect custom flows
      const hasCustomSuffix = clonedConfig.flow_name?.includes('(Custom)');
      const hasCustomPath = clonedConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = clonedConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCustomPath || hasCustomFlag;
      
      
      // Pre-populate wizard with cloned data
      setSelectedFlow({
        name: clonedConfig.flow_name,
        id: clonedConfig.flow_id || clonedConfig.id,
        path: clonedConfig.flow_path,
        tags: clonedConfig.tags || [],
        isCustomFlow: isCustomFlow,
      });
      
      // Copy model and dataset configs
      setModelConfig(modelConfigData);
      setDatasetConfig(datasetConfigData);
      
      // For custom flows, load the flow blocks for cloning
      if (isCustomFlow) {
        // Remove (Copy) suffix if present to get the original flow name
        const originalFlowName = clonedConfig.flow_name.replace(' (Copy)', '');
        loadCustomFlowForCloning(originalFlowName);
      }
      
      // Mark steps as valid
      markStepValid(0, true); // Source selected
      markStepValid(1, isCustomFlow ? false : true); // For custom flows, user needs to save from Build Flow
      markStepValid(2, !!modelConfigData.model);
      markStepValid(3, !!datasetConfigData.data_files);
      
      return;
    }
    
    // If navigated to resume a specific draft
    if (wizardData?.resumeDraft && wizardData?.draftData) {
      setDraftFlow(wizardData.draftData);
      markStepValid(0, true);
      // Mark step 1 as valid if draft has blocks
      if (wizardData.draftData.blocks?.length > 0) {
        markStepValid(1, true);
      }
      return;
    }
    
    // Fresh wizard start
    
    // If there's only one draft, make it easily accessible
    if (drafts.length === 1) {
      setDraftFlow(drafts[0]);
      // Mark step 1 as valid if draft has blocks
      if (drafts[0].blocks?.length > 0) {
        markStepValid(1, true);
      }
    }
  }, []);

  /**
   * Mark a step as valid
   */
  const markStepValid = (stepIndex, isValid) => {
    setStepValidation((prev) => ({
      ...prev,
      [stepIndex]: isValid,
    }));
  };

  /**
   * Load available flows for cloning
   */
  const loadFlows = async () => {
    try {
      setFlowsLoading(true);
      const flows = await flowAPI.listFlows();
      setAvailableFlows(flows);
    } catch (error) {
      console.error('Failed to load flows:', error);
      setErrorMessage('Failed to load flows: ' + error.message);
    } finally {
      setFlowsLoading(false);
    }
  };

  /**
   * Load custom flow for editing (loads existing blocks)
   */
  const loadCustomFlowForEditing = async (flowName) => {
    try {
      
      // Get the flow YAML content from backend
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/flows/${encodeURIComponent(flowName)}/yaml`);
      const flowYamlData = await response.json();
      
      // Set as cloned flow (reusing same state) for editing
      const flowData = {
        blocks: flowYamlData.blocks || [],
        metadata: flowYamlData.metadata || {},
        path: flowYamlData.path,
        isEditing: true, // Flag to indicate we're editing, not creating new
        originalFlowName: flowName, // Track original name for updates
      };
      setClonedFlow(flowData);
      
      // Update selectedFlow with dataset_requirements from flow metadata
      const requiredColumns = flowYamlData.metadata?.required_columns || [];
      setSelectedFlow(prev => ({
        ...prev,
        metadata: flowYamlData.metadata,
        dataset_requirements: {
          required_columns: requiredColumns,
          optional_columns: [],
          description: requiredColumns.length > 0 
            ? `This flow requires the following columns: ${requiredColumns.join(', ')}`
            : 'No specific column requirements for this flow',
        },
      }));
      
      // Mark step 1 as valid if flow has blocks
      if (flowData.blocks?.length > 0) {
        markStepValid(1, true);
      }
      
    } catch (error) {
      // Error: Failed to load custom flow for editing:', error);
      setErrorMessage('Failed to load custom flow: ' + error.message);
    }
  };
  
  /**
   * Load custom flow for cloning (loads existing blocks but creates as new flow)
   */
  const loadCustomFlowForCloning = async (flowName) => {
    try {
      
      // Get the flow YAML content from backend
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/flows/${encodeURIComponent(flowName)}/yaml`);
      const flowYamlData = await response.json();
      
      // Set as cloned flow for modification (will create a new flow with new name)
      const flowData = {
        blocks: flowYamlData.blocks || [],
        metadata: {
          ...flowYamlData.metadata,
          name: `${flowYamlData.metadata?.name || flowName}_copy`,
          description: `Cloned from ${flowYamlData.metadata?.name || flowName}`,
        },
        path: flowYamlData.path,
        isCloning: true, // Flag to indicate we're cloning, not editing
      };
      setClonedFlow(flowData);
      
      // Update selectedFlow with dataset_requirements from flow metadata
      const requiredColumns = flowYamlData.metadata?.required_columns || [];
      setSelectedFlow(prev => ({
        ...prev,
        metadata: flowYamlData.metadata,
        dataset_requirements: {
          required_columns: requiredColumns,
          optional_columns: [],
          description: requiredColumns.length > 0 
            ? `This flow requires the following columns: ${requiredColumns.join(', ')}`
            : 'No specific column requirements for this flow',
        },
      }));
      
      // Mark step 1 as valid if flow has blocks
      if (flowData.blocks?.length > 0) {
        markStepValid(1, true);
      }
      
    } catch (error) {
      // Error: Failed to load custom flow for cloning:', error);
      setErrorMessage('Failed to load custom flow: ' + error.message);
    }
  };
  
  /**
   * Load flow details for cloning
   */
  const loadFlowForClone = async (flowName) => {
    try {
      
      // Load the flow info
      const flowInfo = await flowAPI.getFlowInfo(flowName);
      
      // Get the flow YAML content from backend (this contains the actual blocks)
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/flows/${encodeURIComponent(flowName)}/yaml`);
      const flowYamlData = await response.json();
      
      // Parse blocks from the flow
      const blocks = flowYamlData.blocks || [];
      
      const cloned = {
        blocks: blocks,
        metadata: {
          name: `${flowInfo.name || flowName}_copy`,
          description: `Cloned from ${flowInfo.name || flowName}`,
          version: '1.0.0',
          author: 'SDG Hub User',
          tags: flowInfo.tags || [],
        },
        sourceFlowName: flowName, // Store source flow name for backend to copy prompts
        sourceFlowPath: flowYamlData.path, // Store source flow path
      };
      
      setClonedFlow(cloned);
      setSelectedCloneFlow(flowName);
      
      // Mark step 1 as valid if cloned flow has blocks
      if (cloned.blocks?.length > 0) {
        markStepValid(1, true);
      }
    } catch (error) {
      // Error: Failed to clone flow:', error);
      setErrorMessage('Failed to load flow: ' + error.message);
    }
  };

  /**
   * Handle flow creation completion (from FlowBuilderPage)
   */
  const handleFlowCreated = async (flowData) => {
    try {
      
      // Check if we're editing an existing flow
      const isEditingExisting = clonedFlow?.isEditing && clonedFlow?.originalFlowName;
      
      if (isEditingExisting) {
        // Keep the original flow name - we're updating, not creating
        flowData.metadata = {
          ...flowData.metadata,
          name: clonedFlow.originalFlowName.replace(' (Custom)', ''), // Remove (Custom) suffix if present
        };
      }
      
      // IMPORTANT: Save the flow to backend to get the flow_path
      const flowDataForBackend = {
        metadata: flowData.metadata,
        blocks: flowData.blocks,
        temp_flow_name: flowData.tempFlowName, // For prompt file copying (snake_case for backend)
        source_flow_name: clonedFlow?.sourceFlowName, // For copying prompts from source flow
      };
      
      const saveResponse = await flowAPI.saveCustomFlow(flowDataForBackend);
      
      // Now we have the path from the backend!
      const savedFlowData = {
        ...flowData,
        path: saveResponse.flow_path, // This is critical!
        isEditing: true, // Mark as saved flow for future navigation back
        originalFlowName: flowData.metadata?.name, // Track original name for future saves
      };
      
      setCreatedFlow(savedFlowData);
      
      // Also update clonedFlow so the flow data persists when navigating back to Build Flow step
      setClonedFlow(savedFlowData);
      
      // Get the base name and add "(Custom)" suffix for identification
      const baseName = flowData.metadata?.name || 'Custom Flow';
      const flowNameWithSuffix = isEditingExisting 
        ? (baseName.includes('(Custom)') ? baseName : `${baseName} (Custom)`)
        : `${baseName} (Custom)`;
      
      // Set as selected flow for configuration steps
      // Include dataset_requirements from metadata so DatasetConfigurationStep can use it
      const requiredColumns = flowData.metadata?.required_columns || [];
      setSelectedFlow({
        name: flowNameWithSuffix,
        id: flowData.metadata?.name || 'custom-flow',
        path: saveResponse.flow_path, // Use the path from backend
        tags: flowData.metadata?.tags || [],
        metadata: flowData.metadata,
        dataset_requirements: {
          required_columns: requiredColumns,
          optional_columns: [],
          description: requiredColumns.length > 0 
            ? `This flow requires the following columns: ${requiredColumns.join(', ')}`
            : 'No specific column requirements for this flow',
        },
        isEditingExisting: isEditingExisting, // Flag to track this is an update
        isCustomFlow: true, // Explicit flag to identify custom flows
      });
      
      // DON'T clear drafts here - FlowBuilderPage needs to keep its state
      // We'll clear drafts later when:
      // 1. User completes full configuration (handleWizardSave)
      // 2. User cancels and we save as not_configured (handleWizardClose)
      
      
      // Mark steps as valid
      markStepValid(1, true);
      
      // Note: Don't call onComplete here - that's for final configuration save
      // The wizard will naturally proceed to next step
      
    } catch (error) {
      console.error('Error handling flow creation:', error);
      setErrorMessage('Failed to save flow: ' + error.message);
    }
  };

  // Stable ref for draft ID to prevent re-renders
  const draftIdRef = React.useRef(null);
  
  /**
   * Handle draft changes from FlowBuilderPage
   */
  const handleDraftChange = (draft) => {
    // Don't save drafts if we already have a selectedFlow (flow has been saved)
    if (selectedFlow) {
      return;
    }
    
    // Mark step as valid ONLY if blocks have been added
    if (draft && draft.blocks?.length > 0) {
      markStepValid(1, true);
    } else {
      // No blocks yet - mark as invalid
      markStepValid(1, false);
    }
    
    // Only save if there's actual content (prevent infinite loops)
    if (draft && (draft.blocks?.length > 0 || draft.metadata?.name)) {
      // Load existing drafts
      const existingDrafts = loadAllDrafts();
      
      // Use stable draft ID
      if (!draftIdRef.current) {
        draftIdRef.current = draftFlow?.id || draft.id || `draft_${Date.now()}`;
      }
      const currentDraftId = draftIdRef.current;
      
      const draftToSave = {
        ...draft,
        id: currentDraftId,
        lastModified: new Date().toISOString(),
        name: draft.metadata?.name || 'Unnamed Draft',
      };
      
      // Check if this draft already exists (by ID)
      const existingIndex = existingDrafts.findIndex(d => d.id === currentDraftId);
      
      if (existingIndex >= 0) {
        // Update existing draft
        existingDrafts[existingIndex] = draftToSave;
      } else {
        // Add new draft
        existingDrafts.push(draftToSave);
      }
      
      // Save all drafts (without updating React state to prevent re-renders)
      localStorage.setItem('wizard_drafts', JSON.stringify(existingDrafts));
    }
  };

  // State for Save and Run
  const [isSaveAndRunning, setIsSaveAndRunning] = useState(false);

  /**
   * Handle Save and Run - saves config and triggers generation
   */
  const handleSaveAndRun = async () => {
    try {
      setIsSaveAndRunning(true);
      
      // Check if using direct API key (not env var)
      const apiKey = modelConfig?.api_key || '';
      const usingDirectKey = apiKey && !apiKey.startsWith('env:') && apiKey !== 'EMPTY';
      
      if (usingDirectKey) {
        const confirmed = window.confirm(
          '🔐 SECURITY NOTICE:\n\n' +
          'Your API key will NOT be saved in this configuration for security reasons.\n\n' +
          'When you load this configuration later, you will need to:\n' +
          '1. Re-enter your API key, OR\n' +
          '2. Use environment variables (recommended): Enter "env:YOUR_VAR_NAME" instead\n\n' +
          'Do you want to continue saving and running?'
        );
        
        if (!confirmed) {
          setIsSaveAndRunning(false);
          return;
        }
      }
      
      // Check if we're updating an existing configuration
      const isUpdating = editingConfig && editingConfig.id;
      
      // If updating, delete the old one first (backend doesn't support updates)
      if (isUpdating) {
        try {
          await savedConfigAPI.delete(editingConfig.id);
        } catch (deleteError) {
          console.warn('Failed to delete old config (might not exist):', deleteError);
        }
      }
      
      // Save configuration to backend
      const response = await savedConfigAPI.save({
        flow_name: selectedFlow.name,
        flow_id: selectedFlow.id,
        flow_path: selectedFlow.path || createdFlow?.path || '',
        model_configuration: modelConfig,
        dataset_configuration: datasetConfig,
        dry_run_configuration: dryRunConfig,
        tags: selectedFlow.tags || [],
        status: 'configured',
      });
      
      
      // Clear drafts
      const existingDrafts = loadAllDrafts();
      const currentDraftId = draftIdRef.current || draftFlow?.id;
      const updatedDrafts = existingDrafts.filter(d => {
        if (currentDraftId && d.id === currentDraftId) return false;
        if (d.metadata?.name === selectedFlow.name || d.name === selectedFlow.name) return false;
        return true;
      });
      localStorage.setItem('wizard_drafts', JSON.stringify(updatedDrafts));
      draftIdRef.current = null;
      
      // Clear session state since wizard completed successfully
      clearWizardSessionState();
      
      // Call onComplete with the saved configuration and a flag to run
      if (onComplete) {
        onComplete(response.configuration, { shouldRun: true });
      }
      
      setIsSaveAndRunning(false);
      handleWizardClose(true); // Pass true to indicate successful completion
    } catch (error) {
      console.error('Error in Save and Run:', error);
      setErrorMessage('Failed to save and run: ' + error.message);
      setIsSaveAndRunning(false);
    }
  };

  /**
   * Handle wizard completion - save final configuration
   */
  const handleWizardSave = async () => {
    try {
      // Check if using direct API key (not env var)
      const apiKey = modelConfig?.api_key || '';
      const usingDirectKey = apiKey && !apiKey.startsWith('env:') && apiKey !== 'EMPTY';
      
      if (usingDirectKey) {
        const confirmed = window.confirm(
          '🔐 SECURITY NOTICE:\n\n' +
          'Your API key will NOT be saved in this configuration for security reasons.\n\n' +
          'When you load this configuration later, you will need to:\n' +
          '1. Re-enter your API key, OR\n' +
          '2. Use environment variables (recommended): Enter "env:YOUR_VAR_NAME" instead\n\n' +
          'Do you want to continue saving?'
        );
        
        if (!confirmed) {
          return;
        }
      }
      
      // Check if we're updating an existing configuration
      const isUpdating = editingConfig && editingConfig.id;
      
      // If updating, delete the old one first (backend doesn't support updates)
      if (isUpdating) {
        try {
          await savedConfigAPI.delete(editingConfig.id);
        } catch (deleteError) {
          console.warn('Failed to delete old config (might not exist):', deleteError);
        }
      } else {
      }
      
      // Save configuration to backend
      const response = await savedConfigAPI.save({
        flow_name: selectedFlow.name,
        flow_id: selectedFlow.id,
        flow_path: selectedFlow.path || createdFlow?.path || '',
        model_configuration: modelConfig,
        dataset_configuration: datasetConfig,
        dry_run_configuration: dryRunConfig,
        tags: selectedFlow.tags || [],
        status: 'configured', // Mark as fully configured
      });
      
      
      // Clear any drafts for this flow since it's now fully configured
      const existingDrafts = loadAllDrafts();
      const currentDraftId = draftIdRef.current || draftFlow?.id;
      const updatedDrafts = existingDrafts.filter(d => {
        // Remove current draft by ID
        if (currentDraftId && d.id === currentDraftId) return false;
        // Also remove drafts with same flow name
        if (d.metadata?.name === selectedFlow.name || d.name === selectedFlow.name) return false;
        return true;
      });
      localStorage.setItem('wizard_drafts', JSON.stringify(updatedDrafts));
      draftIdRef.current = null; // Reset
      
      // Show warning if API key was removed
      if (response.warning) {
        alert('⚠️ ' + response.warning);
      }
      
      // Clear session state since wizard completed successfully
      clearWizardSessionState();
      
      // Call onComplete with the saved configuration
      if (onComplete) {
        onComplete(response.configuration);
      }
      
      handleWizardClose(true); // Pass true to indicate successful completion
    } catch (error) {
      console.error('Error saving configuration:', error);
      setErrorMessage('Failed to save configuration: ' + error.message);
    }
  };

  /**
   * Handle "Save and Exit" - saves current changes and exits wizard
   * Only available when editing an existing configuration and changes were made
   */
  const handleSaveAndExit = async () => {
    if (!isEditMode || !hasChanges || !selectedFlow) return;
    
    try {
      // Delete old config first (backend doesn't support updates)
      if (editingConfig?.id) {
        try {
          await savedConfigAPI.delete(editingConfig.id);
        } catch (deleteError) {
          console.warn('Failed to delete old config:', deleteError);
        }
      }
      
      // Determine status based on what's configured
      const isFullyConfigured = modelConfig?.model && datasetConfig?.data_files;
      const status = isFullyConfigured ? 'configured' : 'not_configured';
      
      // Save with current state
      await savedConfigAPI.save({
        flow_name: selectedFlow.name,
        flow_id: selectedFlow.id,
        flow_path: selectedFlow.path || createdFlow?.path || editingConfig?.flow_path || '',
        model_configuration: modelConfig || {},
        dataset_configuration: datasetConfig || {},
        dry_run_configuration: dryRunConfig,
        tags: selectedFlow.tags || editingConfig?.tags || [],
        status: status,
      });
      
      // Clear session state and close
      clearWizardSessionState();
      
      setIsWizardOpen(false);
      if (onCancel) {
        onCancel();
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      setErrorMessage('Failed to save: ' + error.message);
    }
  };

  /**
   * Handle wizard close
   * @param {boolean} completedSuccessfully - If true, wizard completed successfully (session already cleared)
   */
  const handleWizardClose = async (completedSuccessfully = false) => {
    // If completed successfully, session state was already cleared
    // If user is cancelling (clicking Cancel button), clear session state
    // Note: We DON'T clear session state when user navigates away (refresh, clicking other nav)
    // because we want to restore the state when they come back
    
    // When editing and changes were made, save them
    if (isEditMode && hasChanges && selectedFlow) {
      try {
        // Delete old config first (backend doesn't support updates)
        if (editingConfig?.id) {
          try {
            await savedConfigAPI.delete(editingConfig.id);
          } catch (deleteError) {
            console.warn('Failed to delete old config:', deleteError);
          }
        }
        
        // Determine status based on what's configured
        const isFullyConfigured = modelConfig?.model && datasetConfig?.data_files;
        const status = isFullyConfigured ? 'configured' : 'not_configured';
        
        // Save with current state
        await savedConfigAPI.save({
          flow_name: selectedFlow.name,
          flow_id: selectedFlow.id,
          flow_path: selectedFlow.path || createdFlow?.path || editingConfig?.flow_path || '',
          model_configuration: modelConfig || {},
          dataset_configuration: datasetConfig || {},
          dry_run_configuration: dryRunConfig,
          tags: selectedFlow.tags || editingConfig?.tags || [],
          status: status,
        });
      } catch (error) {
        console.error('Failed to save changes on close:', error);
      }
    }
    // For new flows (not editing), save as not_configured if partially filled
    else if (selectedFlow && (!modelConfig.model || !datasetConfig.data_files)) {
      // Check if we're updating an existing configuration
      const isUpdating = editingConfig && editingConfig.id;
      
      // Save as "not_configured" in backend
      try {
        // If updating, delete the old one first (backend doesn't support updates)
        if (isUpdating) {
          try {
            await savedConfigAPI.delete(editingConfig.id);
          } catch (deleteError) {
            console.warn('Failed to delete old config:', deleteError);
          }
        } else {
        }
        
        await savedConfigAPI.save({
          flow_name: selectedFlow.name,
          flow_id: selectedFlow.id,
          flow_path: selectedFlow.path || createdFlow?.path || '',
          model_configuration: modelConfig || {},
          dataset_configuration: datasetConfig || {},
          dry_run_configuration: dryRunConfig,
          tags: selectedFlow.tags || [],
          status: 'not_configured', // Mark as incomplete
        });
        
        // Since we saved this to backend, clear current draft
        const existingDrafts = loadAllDrafts();
        const currentDraftId = draftIdRef.current || draftFlow?.id;
        const updatedDrafts = existingDrafts.filter(d => {
          // Remove current draft by ID
          if (currentDraftId && d.id === currentDraftId) return false;
          // Also remove drafts with same flow name
          if (d.metadata?.name === selectedFlow.name || d.name === selectedFlow.name) return false;
          return true;
        });
        localStorage.setItem('wizard_drafts', JSON.stringify(updatedDrafts));
        draftIdRef.current = null; // Reset
      } catch (error) {
        console.error('Failed to save not_configured status:', error);
      }
    }
    
    setIsWizardOpen(false);
    if (onCancel) {
      onCancel();
    }
  };

  /**
   * Filter flows for cloning
   */
  const filteredFlows = availableFlows.filter(flow => 
    !searchValue || flow.toLowerCase().includes(searchValue.toLowerCase())
  );

  /**
   * Step 1: Choose Flow Source
   */
  const renderSourceSelectionStep = () => (
    <div style={{ 
      padding: '2rem 3rem', 
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#f5f5f5'
    }}>
      <div style={{ marginBottom: '3rem', textAlign: 'center' }}>
        <Title headingLevel="h2" size="2xl" style={{ marginBottom: '12px' }}>
          Choose How to Start
        </Title>
        <p style={{ color: '#6a6e73', fontSize: '16px', maxWidth: '700px', margin: '0 auto' }}>
          Select whether you want to use an existing flow or create a custom flow from scratch.
        </p>
        {selectedCloneFlow && sourceType === 'clone' && (
          <Alert
            variant={AlertVariant.success}
            isInline
            title={`Selected for cloning: ${selectedCloneFlow}`}
            style={{ marginTop: '16px', maxWidth: '700px', margin: '16px auto 0' }}
          />
        )}
      </div>

      <div style={{ 
        width: '100%', 
        maxWidth: '1100px',
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '24px',
        justifyContent: 'center',
      }}>
          <Card 
            isSelectable 
            isSelected={sourceType === 'existing'}
            style={{ 
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: sourceType === 'existing' ? '0 8px 24px rgba(6, 102, 204, 0.25)' : '0 2px 8px rgba(0,0,0,0.1)',
              border: sourceType === 'existing' ? '2px solid #06c' : '2px solid transparent',
              transform: sourceType === 'existing' ? 'translateY(-4px)' : 'none',
              backgroundColor: 'white',
              height: '240px',
              display: 'flex',
              flexDirection: 'column'
            }}
            onClick={() => {
              setSourceType('existing');
              markStepValid(0, true);
            }}
          >
            <CardBody style={{ 
              padding: '40px 24px', 
              textAlign: 'center', 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flex: 1
            }}>
              <CubesIcon style={{ fontSize: '64px', color: '#06c', marginBottom: '20px' }} />
              <div style={{ fontWeight: 600, fontSize: '18px', marginBottom: '10px', color: '#151515' }}>
                Use Existing Flow
              </div>
              <div style={{ color: '#6a6e73', fontSize: '14px', lineHeight: '1.6' }}>
                Select from pre-built flows in the SDG Hub library
              </div>
              <Radio
                isChecked={sourceType === 'existing'}
                name="source-type"
                onChange={() => {
                  setSourceType('existing');
                  markStepValid(0, true);
                }}
                label=""
                id="source-existing"
                aria-label="Use existing flow"
                style={{ position: 'absolute', opacity: 0 }}
              />
            </CardBody>
          </Card>

          <Card 
            isSelectable 
            isSelected={sourceType === 'blank'}
            style={{ 
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: sourceType === 'blank' ? '0 8px 24px rgba(62, 134, 53, 0.25)' : '0 2px 8px rgba(0,0,0,0.1)',
              border: sourceType === 'blank' ? '2px solid #3e8635' : '2px solid transparent',
              transform: sourceType === 'blank' ? 'translateY(-4px)' : 'none',
              backgroundColor: 'white',
              height: '240px',
              display: 'flex',
              flexDirection: 'column'
            }}
            onClick={() => {
              setSourceType('blank');
              markStepValid(0, true);
            }}
          >
            <CardBody style={{ 
              padding: '40px 24px', 
              textAlign: 'center',
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flex: 1
            }}>
              <PlusCircleIcon style={{ fontSize: '64px', color: '#3e8635', marginBottom: '20px' }} />
              <div style={{ fontWeight: 600, fontSize: '18px', marginBottom: '10px', color: '#151515' }}>
                Start from Blank
              </div>
              <div style={{ color: '#6a6e73', fontSize: '14px', lineHeight: '1.6' }}>
                Build a custom flow from scratch using the flow builder
              </div>
              <Radio
                isChecked={sourceType === 'blank'}
                name="source-type"
                onChange={() => {
                  setSourceType('blank');
                  markStepValid(0, true);
                }}
                label=""
                id="source-blank"
                aria-label="Start from blank"
                style={{ position: 'absolute', opacity: 0 }}
              />
            </CardBody>
          </Card>

          <Card 
            isSelectable 
            isSelected={sourceType === 'clone'}
            style={{ 
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: sourceType === 'clone' ? '0 8px 24px rgba(240, 171, 0, 0.25)' : '0 2px 8px rgba(0,0,0,0.1)',
              border: sourceType === 'clone' ? '2px solid #f0ab00' : '2px solid transparent',
              transform: sourceType === 'clone' ? 'translateY(-4px)' : 'none',
              backgroundColor: 'white',
              height: '240px',
              display: 'flex',
              flexDirection: 'column'
            }}
            onClick={() => {
              setSourceType('clone');
              setShowCloneModal(true);
              if (!availableFlows.length) {
                loadFlows();
              }
            }}
          >
            <CardBody style={{ 
              padding: '40px 24px', 
              textAlign: 'center',
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flex: 1
            }}>
              <CopyIcon style={{ fontSize: '64px', color: '#f0ab00', marginBottom: '20px' }} />
              <div style={{ fontWeight: 600, fontSize: '18px', marginBottom: '10px', color: '#151515' }}>
                Clone Existing Flow
              </div>
              <div style={{ color: '#6a6e73', fontSize: '14px', lineHeight: '1.6' }}>
                Create a copy of an existing flow and modify it
              </div>
              <Radio
                isChecked={sourceType === 'clone'}
                name="source-type"
                onChange={() => {
                  setSourceType('clone');
                  setShowCloneModal(true);
                  if (!availableFlows.length) {
                    loadFlows();
                  }
                }}
                label=""
                id="source-clone"
                aria-label="Clone existing flow"
                style={{ position: 'absolute', opacity: 0 }}
              />
            </CardBody>
          </Card>

          {availableDrafts.length > 0 && (
            <Card 
              isSelectable 
              isSelected={sourceType === 'draft'}
              style={{ 
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: sourceType === 'draft' ? '0 8px 24px rgba(139, 67, 221, 0.25)' : '0 2px 8px rgba(0,0,0,0.1)',
                border: sourceType === 'draft' ? '2px solid #8b43dd' : '2px solid transparent',
                transform: sourceType === 'draft' ? 'translateY(-4px)' : 'none',
                backgroundColor: 'white',
                height: '240px',
                display: 'flex',
                flexDirection: 'column'
              }}
              onClick={() => {
                setSourceType('draft');
                setShowDraftModal(true);
              }}
            >
              <CardBody style={{ 
                padding: '40px 24px', 
                textAlign: 'center',
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1
              }}>
                <EditIcon style={{ fontSize: '64px', color: '#8b43dd', marginBottom: '20px' }} />
                <div style={{ fontWeight: 600, fontSize: '18px', marginBottom: '10px', color: '#151515' }}>
                  Continue Draft
                </div>
                <div style={{ color: '#6a6e73', fontSize: '14px', lineHeight: '1.6' }}>
                  Resume work on your saved drafts
                  <br />
                  <Badge style={{ marginTop: '8px' }}>
                    {availableDrafts.length} {availableDrafts.length === 1 ? 'draft' : 'drafts'} available
                  </Badge>
                </div>
                <Radio
                  isChecked={sourceType === 'draft'}
                  name="source-type"
                  onChange={() => {
                    setSourceType('draft');
                    setShowDraftModal(true);
                  }}
                  label=""
                  id="source-draft"
                  aria-label="Continue draft"
                  style={{ position: 'absolute', opacity: 0 }}
                />
              </CardBody>
            </Card>
          )}
      </div>
    </div>
  );

  /**
   * Wizard steps configuration
   */
  const steps = [
    // Step 1: Source Selection
    {
      id: 'source-selection',
      name: 'Choose Source',
      component: renderSourceSelectionStep(),
      enableNext: stepValidation[0],
    },
    
    // Step 2a: Select Existing Flow (only shown if sourceType === 'existing')
    ...(sourceType === 'existing' ? [{
      id: 'select-existing',
      name: 'Select Flow',
      component: (
        <div style={{ padding: '1.5rem 2.5rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          {selectedFlow && editingConfig ? (
            <Alert
              variant={AlertVariant.success}
              isInline
              title="Flow already selected"
              style={{ marginBottom: '20px', flexShrink: 0 }}
            >
              Currently editing: <strong>{selectedFlow.name}</strong>. Click Next to proceed to configuration.
            </Alert>
          ) : (
            <Alert
              variant={AlertVariant.info}
              isInline
              title="Choose a flow"
              style={{ marginBottom: '20px', flexShrink: 0 }}
            >
              Browse and select a flow from the SDG Hub library.
            </Alert>
          )}
          <div style={{ flex: 1, minHeight: 0 }}>
            <FlowSelectionStep
              selectedFlow={selectedFlow}
              onFlowSelect={(flow) => {
                setSelectedFlow(flow);
                markStepValid(1, true);
              }}
              isImported={!!editingConfig}
              onError={setErrorMessage}
            />
          </div>
        </div>
      ),
      enableNext: stepValidation[1],
      canJumpTo: stepValidation[0],
    }] : []),
    
    // Step 2b: Build Custom Flow (only shown if sourceType !== 'existing')
    ...(sourceType !== 'existing' && sourceType !== null ? [{
      id: 'build-flow',
      name: 'Build Flow',
      component: (
        <div 
          style={{ 
            height: '100%', 
            width: '100%',
            overflow: 'auto',
            padding: 0
          }}
        >
          <FlowBuilderPage
            key={`builder-${sourceType}-${selectedCloneFlow || selectedDraftId || editingConfig?.id || createdFlow?.metadata?.name || 'blank'}`}
            initialFlow={
              // Priority: 1) Already created/saved flow, 2) Cloned flow, 3) Draft flow, 4) null for blank
              createdFlow ? createdFlow :
              clonedFlow ? clonedFlow :
              sourceType === 'draft' ? draftFlow :
              null
            }
            onBack={() => {
              // User clicked back in FlowBuilderPage - reset source selection
              setSourceType(null);
              setClonedFlow(null);
              setDraftFlow(null);
              setCreatedFlow(null);
              setSelectedFlow(null);
              markStepValid(0, false);
              markStepValid(1, false);
            }}
            onSave={handleFlowCreated}
            onDraftChange={handleDraftChange}
            triggerSave={flowBuilderSaveRef}
            autoSaveOnNext={(info) => setFlowBuilderSaveInfo(info)}
          />
        </div>
      ),
      enableNext: !!selectedFlow,
      // Custom footer for build-flow step with Save Changes button
      customFooter: true,  
      canJumpTo: stepValidation[0],
    }] : []),
    
    // Step 3: Model Configuration
    {
      id: 'model-configuration',
      name: 'Configure Model',
      component: (
        <div style={{ padding: '1.5rem 2.5rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Alert
            variant={AlertVariant.info}
            isInline
            title="Model settings"
            style={{ marginBottom: '20px', flexShrink: 0 }}
          >
            Configure the language model that will be used for generation tasks.
          </Alert>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ModelConfigurationStep
              selectedFlow={selectedFlow}
              modelConfig={modelConfig}
              importedConfig={null}
              onConfigChange={(config) => {
                setModelConfig(config);
                markStepValid(2, config.model ? true : false);
              }}
              onError={setErrorMessage}
            />
          </div>
        </div>
      ),
      enableNext: stepValidation[2],
      canJumpTo: stepValidation[1],
    },
    
    // Step 4: Dataset Configuration
    {
      id: 'dataset-configuration',
      name: 'Configure Dataset',
      component: (
        <div style={{ padding: '1.5rem 2.5rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Alert
            variant={AlertVariant.info}
            isInline
            title="Dataset configuration"
            style={{ marginBottom: '20px', flexShrink: 0 }}
          >
            Load and configure the dataset that will be used as input for your flow.
          </Alert>
          <div style={{ flex: 1, minHeight: 0 }}>
            <DatasetConfigurationStep
              selectedFlow={selectedFlow}
              datasetConfig={datasetConfig}
              importedConfig={null}
              onConfigChange={(config) => {
                setDatasetConfig(config);
                markStepValid(3, config.data_files ? true : false);
              }}
              onError={setErrorMessage}
            />
          </div>
        </div>
      ),
      enableNext: stepValidation[3],
      canJumpTo: stepValidation[1] && stepValidation[2],
    },
    
    // Step 5: Dry Run
    {
      id: 'dry-run-settings',
      name: 'Dry Run',
      component: (
        <div style={{ height: '100%' }}>
          <DryRunSettingsStep
            dryRunConfig={dryRunConfig}
            onConfigChange={setDryRunConfig}
            selectedFlow={selectedFlow}
            modelConfig={modelConfig}
            datasetConfig={datasetConfig}
          />
        </div>
      ),
      enableNext: true,
      canJumpTo: stepValidation[1] && stepValidation[2] && stepValidation[3],
    },
    
    // Step 6: Review & Confirm
    {
      id: 'review',
      name: 'Review & Confirm',
      component: (
        <div style={{ padding: '1.5rem 2.5rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Alert
            variant={AlertVariant.info}
            isInline
            title="Review your configuration"
            style={{ marginBottom: '20px', flexShrink: 0 }}
          >
            Review all settings before saving. You can go back to modify any step if needed.
          </Alert>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ReviewStep
              selectedFlow={selectedFlow}
              modelConfig={modelConfig}
              datasetConfig={datasetConfig}
              onError={setErrorMessage}
            />
          </div>
        </div>
      ),
      enableNext: true,
      canJumpTo: stepValidation[1] && stepValidation[2] && stepValidation[3],
      nextButtonText: 'Save Configuration',
      isReviewStep: true, // Flag to identify review step for custom footer
    },
  ];

  if (!isWizardOpen) {
    return null;
  }

  /**
   * Draft Selection Modal
   */
  const renderDraftModal = () => (
    <Modal
      variant={ModalVariant.medium}
      title="Select Draft to Continue"
      isOpen={showDraftModal}
      onClose={() => {
        setShowDraftModal(false);
        if (!selectedDraftId) {
          setSourceType(null);
          markStepValid(0, false);
        }
      }}
      actions={[
        <Button
          key="select"
          variant="primary"
          onClick={() => {
            if (selectedDraftId) {
              const draft = availableDrafts.find(d => d.id === selectedDraftId);
              if (draft) {
                setDraftFlow(draft);
                setShowDraftModal(false);
                markStepValid(0, true);
                // Mark step 1 as valid if draft has blocks
                if (draft.blocks?.length > 0) {
                  markStepValid(1, true);
                } else {
                  markStepValid(1, false);
                }
              }
            }
          }}
          isDisabled={!selectedDraftId}
        >
          Continue with Draft
        </Button>,
        <Button 
          key="cancel" 
          variant="link" 
          onClick={() => {
            setShowDraftModal(false);
            setSourceType(null);
            setSelectedDraftId(null);
            markStepValid(0, false);
          }}
        >
          Cancel
        </Button>
      ]}
    >
      <div style={{ padding: '16px 0' }}>
        {availableDrafts.length === 0 ? (
          <EmptyState>
            <EmptyStateHeader 
              titleText="No drafts found" 
              icon={<EmptyStateIcon icon={EditIcon} />}
              headingLevel="h4"
            />
            <EmptyStateBody>
              You don't have any saved drafts yet. Start building a flow and it will be auto-saved as a draft.
            </EmptyStateBody>
          </EmptyState>
        ) : (
          <List isPlain isBordered style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {availableDrafts.map((draft) => (
              <ListItem
                key={draft.id}
                onClick={() => setSelectedDraftId(draft.id)}
                style={{ 
                  cursor: 'pointer',
                  backgroundColor: selectedDraftId === draft.id ? '#e7f1fa' : 'transparent',
                  padding: '12px 16px',
                  transition: 'background-color 0.2s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <div>
                    <div style={{ fontWeight: selectedDraftId === draft.id ? 600 : 400 }}>
                      {draft.name || 'Unnamed Draft'}
                    </div>
                    <div style={{ fontSize: '12px', color: '#6a6e73', marginTop: '4px' }}>
                      {draft.blocks?.length || 0} blocks • Last modified: {new Date(draft.lastModified).toLocaleString()}
                    </div>
                  </div>
                  {selectedDraftId === draft.id && <Badge isRead>Selected</Badge>}
                </div>
              </ListItem>
            ))}
          </List>
        )}
      </div>
    </Modal>
  );
  
  /**
   * Clone Flow Modal
   */
  const renderCloneModal = () => (
    <Modal
      variant={ModalVariant.medium}
      title="Select Flow to Clone"
      isOpen={showCloneModal}
      onClose={() => {
        setShowCloneModal(false);
        if (!selectedCloneFlow) {
          setSourceType(null);
          markStepValid(0, false);
        }
      }}
      actions={[
        <Button
          key="select"
          variant="primary"
          onClick={() => {
            if (selectedCloneFlow && clonedFlow) {
              setShowCloneModal(false);
              markStepValid(0, true);
              markStepValid(1, false); // Build step not complete yet
            }
          }}
          isDisabled={!selectedCloneFlow || !clonedFlow}
        >
          Select Flow
        </Button>,
        <Button 
          key="cancel" 
          variant="link" 
          onClick={() => {
            setShowCloneModal(false);
            setSourceType(null);
            setSelectedCloneFlow(null);
            setClonedFlow(null);
            markStepValid(0, false);
          }}
        >
          Cancel
        </Button>
      ]}
    >
      <div style={{ padding: '16px 0' }}>
        <SearchInput
          placeholder="Search flows to clone..."
          value={searchValue}
          onChange={(_event, value) => setSearchValue(value)}
          onClear={() => setSearchValue('')}
          style={{ marginBottom: '16px' }}
        />
        
        {flowsLoading ? (
          <div style={{ textAlign: 'center', padding: '32px' }}>
            <Spinner size="lg" />
          </div>
        ) : filteredFlows.length === 0 ? (
          <EmptyState>
            <EmptyStateHeader 
              titleText="No flows found" 
              icon={<EmptyStateIcon icon={CubesIcon} />}
              headingLevel="h4"
            />
            <EmptyStateBody>
              {searchValue ? 'Try adjusting your search' : 'No flows available to clone'}
            </EmptyStateBody>
          </EmptyState>
        ) : (
          <List isPlain isBordered style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {filteredFlows.map((flow) => (
              <ListItem
                key={flow}
                onClick={() => loadFlowForClone(flow)}
                style={{ 
                  cursor: 'pointer',
                  backgroundColor: selectedCloneFlow === flow ? '#e7f1fa' : 'transparent',
                  padding: '12px 16px',
                  transition: 'background-color 0.2s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <span style={{ fontWeight: selectedCloneFlow === flow ? 600 : 400 }}>{flow}</span>
                  {selectedCloneFlow === flow && <Badge isRead>Selected</Badge>}
                </div>
              </ListItem>
            ))}
          </List>
        )}
      </div>
    </Modal>
  );

  return (
    <>
      {/* Page Header */}
      <PageSection variant="light" style={{ paddingBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <Title headingLevel="h1" size="2xl">Configure Flow</Title>
            <p style={{ 
              marginTop: '8px', 
              color: '#6a6e73',
              fontSize: '14px'
            }}>
              {sourceType === null 
                ? 'Choose how you want to create or configure your flow'
                : sourceType === 'existing'
                ? 'Select and configure an existing flow'
                : 'Build and configure your custom flow'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            {/* Save and Exit button - only show when editing and changes were made */}
            {isEditMode && hasChanges && (
              <Button
                variant="primary"
                onClick={handleSaveAndExit}
              >
                Save and Exit
              </Button>
            )}
            <Button
              variant="secondary"
              onClick={() => {
                // User explicitly cancelled - clear session state
                clearWizardSessionState();
                handleWizardClose(true);
              }}
            >
              Cancel & Return to Flows
            </Button>
          </div>
        </div>
      </PageSection>

      <PageSection style={{ padding: 0, height: 'calc(100vh - 140px)', display: 'flex', flexDirection: 'column', paddingBottom: '20px' }}>
        {errorMessage && (
          <Alert
            variant={AlertVariant.danger}
            title="Error"
            isInline
            actionClose={<Button variant="plain" onClick={() => setErrorMessage(null)}>×</Button>}
            style={{ margin: '16px 24px', flexShrink: 0 }}
          >
            {errorMessage}
          </Alert>
        )}
        
      <div style={{ flex: 1, overflow: 'hidden', marginBottom: '20px' }}>
        <Wizard
          key={`wizard-${editingConfig?.id || wizardData?.draftData?.id || 'new'}`}
          onSave={handleWizardSave}
          onStepChange={(event, currentStep) => {
            // Track current step for session persistence
            if (currentStep?.id) {
              setCurrentStepId(currentStep.id);
            }
          }}
          height="100%"
          style={{ paddingBottom: '20px' }}
          startIndex={(() => {
            if (!initialStepId) {
              return 1;
            }
            
            
            const stepIndex = steps.findIndex(s => s.id === initialStepId);
            
            if (stepIndex < 0) {
              return 1;
            }
            
            // PatternFly v5 Wizard uses 1-based indexing for startIndex
            return stepIndex + 1;
          })()}
        >
          {steps.map((step, index) => (
            <WizardStep
              key={step.id}
              id={step.id}
              name={step.name}
              footer={step.isReviewStep ? (
                <WizardFooterWrapper>
                  <Button 
                    variant="secondary" 
                    onClick={() => {
                      // Go back to previous step
                      const wizard = document.querySelector('.pf-v5-c-wizard');
                      const backBtn = wizard?.querySelector('.pf-v5-c-wizard__footer-cancel')?.previousElementSibling;
                      if (backBtn) backBtn.click();
                    }}
                  >
                    Back
                  </Button>
                  <Button
                    variant="primary"
                    icon={isSaveAndRunning ? <Spinner size="sm" /> : <PlayIcon />}
                    onClick={handleSaveAndRun}
                    isDisabled={isSaveAndRunning}
                    isLoading={isSaveAndRunning}
                  >
                    {isSaveAndRunning ? 'Saving...' : 'Save and Run'}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={handleWizardSave}
                    isDisabled={isSaveAndRunning}
                  >
                    Save to Flows List
                  </Button>
                </WizardFooterWrapper>
              ) : step.customFooter ? (
                <BuildFlowFooter 
                  flowBuilderSaveInfo={flowBuilderSaveInfo} 
                  selectedFlow={selectedFlow}
                />
              ) : {
                isNextDisabled: !step.enableNext,
                isCancelHidden: true,
                nextButtonText: step.nextButtonText,
              }}
            >
              {step.component}
            </WizardStep>
          ))}
        </Wizard>
        </div>
      </PageSection>

      {/* Clone Flow Modal */}
      {renderCloneModal()}
      
      {/* Draft Selection Modal */}
      {renderDraftModal()}
    </>
  );
};

export default UnifiedFlowWizard;

