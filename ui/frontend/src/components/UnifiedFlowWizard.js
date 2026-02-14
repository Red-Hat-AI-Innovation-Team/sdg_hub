import React, { useState, useEffect, useReducer } from 'react';
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
import PDFPreprocessingStep from './steps/PDFPreprocessingStep';
import DryRunSettingsStep from './steps/DryRunSettingsStep';
import ReviewStep from './steps/ReviewStep';
import FlowBuilderPage from './flowCreator/FlowBuilderPage';
import { flowAPI, savedConfigAPI, API_BASE_URL } from '../services/api';
import { ExecutionProvider } from '../contexts/ExecutionContext';

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

/**
 * Custom Footer for Review step with simplified Save options
 * "Run Now" = Save + Execute immediately
 * "Save for Later" = Save only, no execution
 */
const ReviewFooter = ({ onSaveAndRun, onSave, isSaveAndRunning }) => {
  const { goToPrevStep } = useWizardContext();
  
  return (
    <WizardFooterWrapper>
      {/* Back Button */}
      <Button variant="secondary" onClick={goToPrevStep}>
        Back
      </Button>
      
      {/* Primary Action: Run Now */}
      <Button
        variant="primary"
        icon={isSaveAndRunning ? <Spinner size="sm" /> : <PlayIcon />}
        onClick={onSaveAndRun}
        isDisabled={isSaveAndRunning}
        isLoading={isSaveAndRunning}
        style={{ marginLeft: '1rem' }}
      >
        {isSaveAndRunning ? 'Starting...' : 'Run Now'}
      </Button>
      
      {/* Secondary Action: Save for Later */}
      <Button
        variant="tertiary"
        onClick={onSave}
        isDisabled={isSaveAndRunning}
        style={{ marginLeft: '0.5rem' }}
      >
        Save for Later
      </Button>
    </WizardFooterWrapper>
  );
};

/**
 * Custom Footer for Dry Run step with navigation guard
 * Shows confirmation when trying to navigate away during an active dry run
 */
const DryRunFooter = ({ isDryRunActive, onShowExitModal, pendingNavigation, onClearPendingNavigation }) => {
  const { goToNextStep, goToPrevStep } = useWizardContext();
  
  // When dry run is stopped and there's a pending navigation, execute it
  React.useEffect(() => {
    if (!isDryRunActive && pendingNavigation) {
      if (pendingNavigation === 'next') {
        goToNextStep();
      } else {
        goToPrevStep();
      }
      onClearPendingNavigation?.();
    }
  }, [isDryRunActive, pendingNavigation, goToNextStep, goToPrevStep, onClearPendingNavigation]);
  
  const handleNavigation = (direction) => {
    if (isDryRunActive) {
      onShowExitModal(direction);
    } else {
      if (direction === 'next') {
        goToNextStep();
      } else {
        goToPrevStep();
      }
    }
  };
  
  return (
    <WizardFooterWrapper>
      <Button variant="secondary" onClick={() => handleNavigation('back')}>
        Back
      </Button>
      <Button variant="primary" onClick={() => handleNavigation('next')}>
        Next
      </Button>
    </WizardFooterWrapper>
  );
};

// Session storage key for wizard state persistence
const WIZARD_SESSION_KEY = 'wizard_session_state';

/**
 * Determine if we're starting a specific wizard context (edit/clone/draft)
 * Returns a context ID if so, null for new wizards
 */
const getWizardContextId = (editingConfig, wizardData) => {
  if (editingConfig?.id) {
    return `edit_${editingConfig.id}`;
  }
  if (wizardData?.clonedConfig?.id) {
    return `clone_${wizardData.clonedConfig.id}`;
  }
  if (wizardData?.draftData?.id) {
    return `draft_${wizardData.draftData.id}`;
  }
  if (wizardData?.resumeDraft?.id) {
    return `draft_${wizardData.resumeDraft.id}`;
  }
  // New wizard - no specific context
  return null;
};

/**
 * Load saved wizard session state from sessionStorage
 * - For edit/clone/draft: only returns state if context matches
 * - For new wizard: returns state only if it was also a new wizard (no context)
 */
const loadWizardSessionState = (contextId) => {
  try {
    const saved = sessionStorage.getItem(WIZARD_SESSION_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      
      // For specific contexts (edit/clone/draft), only restore matching state
      if (contextId) {
        if (parsed.wizardContextId === contextId) {
          return parsed;
        }
        return null;
      }
      
      // For new wizard (no contextId), only restore if saved state was also for new wizard
      if (!contextId && !parsed.wizardContextId) {
        return parsed;
      }
      
      return null;
    }
  } catch (error) {
    console.error('Failed to load wizard session state:', error);
    sessionStorage.removeItem(WIZARD_SESSION_KEY);
  }
  return null;
};

/**
 * Clear wizard session state AND all PDF preprocessing states
 */
const clearWizardSessionState = () => {
  sessionStorage.removeItem(WIZARD_SESSION_KEY);
  
  // Also clear all PDF preprocessing states (they start with 'sdg_hub_wizard_pdf_state_')
  const keysToRemove = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (key && key.startsWith('sdg_hub_wizard_pdf_state_')) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach(key => sessionStorage.removeItem(key));
};

/**
 * Generic reducer for grouped state.
 * Supports SET_FIELD (single field, including functional updates),
 * MERGE (spread multiple fields), and RESET (replace entire state).
 */
const groupReducer = (state, action) => {
  switch (action.type) {
    case 'SET_FIELD': {
      const value = typeof action.value === 'function'
        ? action.value(state[action.field])
        : action.value;
      return { ...state, [action.field]: value };
    }
    case 'MERGE':
      return { ...state, ...action.payload };
    case 'RESET':
      return action.payload !== undefined ? action.payload : state;
    default:
      return state;
  }
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
  // Get wizard context ID (edit/clone/draft or null for new wizard)
  const wizardContextId = React.useMemo(() => {
    return getWizardContextId(editingConfig, wizardData);
  }, [editingConfig, wizardData]);
  
  // Try to restore session state - only if context matches
  const savedSessionState = React.useMemo(() => {
    return loadWizardSessionState(wizardContextId);
  }, [wizardContextId]);
  
  // Check if we should restore session state
  // Restore if we have saved state AND no explicit intent that overrides it
  const shouldRestoreSession = savedSessionState && 
    !wizardData?.sourceType && // Not explicitly setting source type
    !wizardData?.isCloning;    // Not explicitly cloning
  
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
  
  // Flow state (useReducer group: sourceType, selectedFlow, createdFlow, clonedFlow, draftFlow)
  const [flowState, flowDispatch] = useReducer(groupReducer, {
    sourceType: initialSourceType,
    selectedFlow: shouldRestoreSession ? savedSessionState.selectedFlow : null,
    createdFlow: shouldRestoreSession ? savedSessionState.createdFlow : null,
    clonedFlow: shouldRestoreSession ? savedSessionState.clonedFlow : null,
    draftFlow: shouldRestoreSession ? savedSessionState.draftFlow : null,
  });
  const { sourceType, selectedFlow, createdFlow, clonedFlow, draftFlow } = flowState;
  const setSourceType = (v) => flowDispatch({ type: 'SET_FIELD', field: 'sourceType', value: v });
  const setSelectedFlow = (v) => flowDispatch({ type: 'SET_FIELD', field: 'selectedFlow', value: v });
  const setCreatedFlow = (v) => flowDispatch({ type: 'SET_FIELD', field: 'createdFlow', value: v });
  const setClonedFlow = (v) => flowDispatch({ type: 'SET_FIELD', field: 'clonedFlow', value: v });
  const setDraftFlow = (v) => flowDispatch({ type: 'SET_FIELD', field: 'draftFlow', value: v });
  const resetFlowState = () => flowDispatch({ type: 'RESET', payload: {
    sourceType: null, selectedFlow: null, createdFlow: null, clonedFlow: null, draftFlow: null,
  }});
  
  // Config state (useReducer group: modelConfig, datasetConfig, dryRunConfig)
  const [configState, configDispatch] = useReducer(groupReducer, {
    modelConfig: shouldRestoreSession && savedSessionState.modelConfig ? savedSessionState.modelConfig : {},
    datasetConfig: shouldRestoreSession && savedSessionState.datasetConfig ? savedSessionState.datasetConfig : {},
    dryRunConfig: shouldRestoreSession && savedSessionState.dryRunConfig ? savedSessionState.dryRunConfig : {
      sample_size: 2,
      enable_time_estimation: true,
      max_concurrency: 10,
    },
  });
  const { modelConfig, datasetConfig, dryRunConfig } = configState;
  const setModelConfig = (v) => configDispatch({ type: 'SET_FIELD', field: 'modelConfig', value: v });
  const setDatasetConfig = (v) => configDispatch({ type: 'SET_FIELD', field: 'datasetConfig', value: v });
  const setDryRunConfig = (v) => configDispatch({ type: 'SET_FIELD', field: 'dryRunConfig', value: v });

  // Memoized context value for ExecutionContext (eliminates prop drilling of read-only config)
  const executionContextValue = React.useMemo(() => ({
    selectedFlow,
    modelConfig,
    datasetConfig,
  }), [selectedFlow, modelConfig, datasetConfig]);

  // Dataset state (useReducer group: datasetSourceType, pdfPreprocessingState, pdfDatasetInfo)
  const [datasetState, datasetDispatch] = useReducer(groupReducer, {
    datasetSourceType: shouldRestoreSession && savedSessionState.datasetSourceType ? savedSessionState.datasetSourceType : 'none',
    pdfPreprocessingState: shouldRestoreSession && savedSessionState.pdfPreprocessingState ? savedSessionState.pdfPreprocessingState : null,
    pdfDatasetInfo: shouldRestoreSession && savedSessionState.pdfDatasetInfo ? savedSessionState.pdfDatasetInfo : null,
  });
  const { datasetSourceType, pdfPreprocessingState, pdfDatasetInfo } = datasetState;
  const setDatasetSourceType = (v) => datasetDispatch({ type: 'SET_FIELD', field: 'datasetSourceType', value: v });
  const setPdfPreprocessingState = (v) => datasetDispatch({ type: 'SET_FIELD', field: 'pdfPreprocessingState', value: v });
  const setPdfDatasetInfo = (v) => datasetDispatch({ type: 'SET_FIELD', field: 'pdfDatasetInfo', value: v });

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
  
  // Clone modal state (useReducer group: availableFlows, flowsLoading, searchValue, selectedCloneFlow, showCloneModal)
  const [cloneModalState, cloneModalDispatch] = useReducer(groupReducer, {
    availableFlows: [],
    flowsLoading: false,
    searchValue: '',
    selectedCloneFlow: shouldRestoreSession ? savedSessionState.selectedCloneFlow : null,
    showCloneModal: false,
  });
  const { availableFlows, flowsLoading, searchValue, selectedCloneFlow, showCloneModal } = cloneModalState;
  const setAvailableFlows = (v) => cloneModalDispatch({ type: 'SET_FIELD', field: 'availableFlows', value: v });
  const setFlowsLoading = (v) => cloneModalDispatch({ type: 'SET_FIELD', field: 'flowsLoading', value: v });
  const setSearchValue = (v) => cloneModalDispatch({ type: 'SET_FIELD', field: 'searchValue', value: v });
  const setSelectedCloneFlow = (v) => cloneModalDispatch({ type: 'SET_FIELD', field: 'selectedCloneFlow', value: v });
  const setShowCloneModal = (v) => cloneModalDispatch({ type: 'SET_FIELD', field: 'showCloneModal', value: v });
  
  // Draft modal state (useReducer group: availableDrafts, showDraftModal, selectedDraftId)
  const [draftModalState, draftModalDispatch] = useReducer(groupReducer, {
    availableDrafts: [],
    showDraftModal: false,
    selectedDraftId: shouldRestoreSession ? savedSessionState.selectedDraftId : null,
  });
  const { availableDrafts, showDraftModal, selectedDraftId } = draftModalState;
  const setAvailableDrafts = (v) => draftModalDispatch({ type: 'SET_FIELD', field: 'availableDrafts', value: v });
  const setShowDraftModal = (v) => draftModalDispatch({ type: 'SET_FIELD', field: 'showDraftModal', value: v });
  const setSelectedDraftId = (v) => draftModalDispatch({ type: 'SET_FIELD', field: 'selectedDraftId', value: v });
  
  // Validation state - restore from session if available
  // Mark step 0 as valid if sourceType was pre-set via wizardData (e.g., from Home page flow click)
  const [stepValidation, setStepValidation] = useState(
    shouldRestoreSession && savedSessionState.stepValidation ? savedSessionState.stepValidation : {
      0: !!wizardData?.sourceType || false, // Source selection - auto-valid if pre-set
      1: false, // Flow selection/creation
      2: false, // Model configuration
      3: false, // Dataset source selection
      4: false, // Dataset configuration
      5: true,  // Dry run (optional)
      6: true,  // Review
    }
  );
  
  // When dataset source type changes, reset dataset config validation
  // (so the user must re-configure the dataset for the new source)
  useEffect(() => {
    if (datasetSourceType !== 'none') {
      // Reset dataset configuration validation when source changes
      setStepValidation(prev => ({ ...prev, 4: false }));
      // If switching away from pdf, clear pdf-specific state
      if (datasetSourceType !== 'pdf') {
        setPdfDatasetInfo(null);
      }
    }
  }, [datasetSourceType]);

  // Dry run state (useReducer group: isDryRunActive, showDryRunExitModal, pendingNavigation)
  const [dryRunState, dryRunDispatch] = useReducer(groupReducer, {
    isDryRunActive: false,
    showDryRunExitModal: false,
    pendingNavigation: null,
  });
  const { isDryRunActive, showDryRunExitModal, pendingNavigation } = dryRunState;
  const setIsDryRunActive = (v) => dryRunDispatch({ type: 'SET_FIELD', field: 'isDryRunActive', value: v });
  const setShowDryRunExitModal = (v) => dryRunDispatch({ type: 'SET_FIELD', field: 'showDryRunExitModal', value: v });
  const setPendingNavigation = (v) => dryRunDispatch({ type: 'SET_FIELD', field: 'pendingNavigation', value: v });
  
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
      wizardContextId, // Include context ID for matching on restore (null for new wizards)
      sourceType,
      selectedFlow,
      createdFlow,
      clonedFlow,
      draftFlow,
      modelConfig,
      datasetConfig,
      dryRunConfig,
      datasetSourceType,
      pdfPreprocessingState,
      pdfDatasetInfo,
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
    wizardContextId,
    sourceType,
    selectedFlow,
    createdFlow,
    clonedFlow,
    draftFlow,
    modelConfig,
    datasetConfig,
    dryRunConfig,
    datasetSourceType,
    pdfPreprocessingState,
    pdfDatasetInfo,
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
      // Note: Custom flows may have "(Custom)" or "(Copy)" suffix, or be in custom_flows directory
      const hasCustomSuffix = editingConfig.flow_name?.includes('(Custom)');
      const hasCopySuffix = editingConfig.flow_name?.includes('(Copy)'); // Templates loaded get "(Copy)" suffix
      const hasCustomPath = editingConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = editingConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCopySuffix || hasCustomPath || hasCustomFlag;
      
      console.log('UnifiedFlowWizard: Editing config detection:', {
        flowName: editingConfig.flow_name,
        flowPath: editingConfig.flow_path,
        hasCustomSuffix,
        hasCopySuffix,
        hasCustomPath,
        hasCustomFlag,
        isCustomFlow,
      });
      
      
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
      markStepValid(3, hasDataset); // Dataset source
      markStepValid(4, hasDataset); // Dataset configuration
      
      return;
    }
    
    // If navigated to clone a configuration
    if (wizardData?.isCloning && wizardData?.clonedConfig) {
      const clonedConfig = wizardData.clonedConfig;
      const modelConfigData = clonedConfig.model_configuration || clonedConfig.model_config || {};
      const datasetConfigData = clonedConfig.dataset_configuration || clonedConfig.dataset_config || {};
      
      // Multiple ways to detect custom flows
      const hasCustomSuffix = clonedConfig.flow_name?.includes('(Custom)');
      const hasCopySuffix = clonedConfig.flow_name?.includes('(Copy)');
      const hasCustomPath = clonedConfig.flow_path?.includes('custom_flows');
      const hasCustomFlag = clonedConfig.isCustomFlow === true;
      const isCustomFlow = hasCustomSuffix || hasCopySuffix || hasCustomPath || hasCustomFlag;
      
      
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
      markStepValid(3, !!datasetConfigData.data_files); // Dataset source
      markStepValid(4, !!datasetConfigData.data_files); // Dataset configuration
      
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
      if (!response.ok) {
        throw new Error(`Failed to load flow YAML: ${response.status} ${response.statusText}`);
      }
      const flowYamlData = await response.json();
      
      console.log('loadCustomFlowForEditing: Loaded flow YAML for', flowName);
      console.log('loadCustomFlowForEditing: Full response keys:', Object.keys(flowYamlData));
      console.log('loadCustomFlowForEditing: visualNodes count:', flowYamlData.visualNodes?.length);
      console.log('loadCustomFlowForEditing: visualEdges count:', flowYamlData.visualEdges?.length);
      console.log('loadCustomFlowForEditing: blocks count:', flowYamlData.blocks?.length);
      
      // Set as cloned flow (reusing same state) for editing
      // Include visualNodes and visualEdges if they were saved with the flow
      // If not available, set to null to trigger regeneration from blocks in FlowBuilderPage
      const hasVisualData = flowYamlData.visualNodes && flowYamlData.visualNodes.length > 0;
      const flowData = {
        blocks: flowYamlData.blocks || [],
        metadata: flowYamlData.metadata || {},
        path: flowYamlData.path,
        visualNodes: hasVisualData ? flowYamlData.visualNodes : null, // null triggers regeneration
        visualEdges: hasVisualData ? flowYamlData.visualEdges : null, // null triggers regeneration
        needsVisualRegeneration: !hasVisualData, // Flag to indicate we need to regenerate from blocks
        isEditing: true, // Flag to indicate we're editing, not creating new
        originalFlowName: flowName, // Track original name for updates
      };
      
      console.log('loadCustomFlowForEditing: Setting clonedFlow with hasVisualData:', hasVisualData);
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
      console.log('loadCustomFlowForCloning: Loading flow', flowName);
      
      // Get the flow YAML content from backend
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/flows/${encodeURIComponent(flowName)}/yaml`);
      if (!response.ok) {
        throw new Error(`Failed to load flow YAML: ${response.status} ${response.statusText}`);
      }
      const flowYamlData = await response.json();
      
      console.log('loadCustomFlowForCloning: Full response keys:', Object.keys(flowYamlData));
      console.log('loadCustomFlowForCloning: visualNodes count:', flowYamlData.visualNodes?.length);
      console.log('loadCustomFlowForCloning: visualEdges count:', flowYamlData.visualEdges?.length);
      console.log('loadCustomFlowForCloning: blocks count:', flowYamlData.blocks?.length);
      
      // Check if we have visual layout data saved with the flow
      const hasVisualData = flowYamlData.visualNodes && flowYamlData.visualNodes.length > 0;
      
      // Set as cloned flow for modification (will create a new flow with new name)
      // Include visual nodes/edges if available, otherwise they'll be regenerated from blocks
      const flowData = {
        blocks: flowYamlData.blocks || [],
        metadata: {
          ...flowYamlData.metadata,
          name: `${flowYamlData.metadata?.name || flowName}_copy`,
          description: `Cloned from ${flowYamlData.metadata?.name || flowName}`,
        },
        path: flowYamlData.path,
        visualNodes: hasVisualData ? flowYamlData.visualNodes : null, // Include visual layout
        visualEdges: hasVisualData ? flowYamlData.visualEdges : null, // Include visual edges
        needsVisualRegeneration: !hasVisualData, // Flag to indicate we need to regenerate from blocks
        isCloning: true, // Flag to indicate we're cloning, not editing
        sourceFlowName: flowName, // Track source flow for copying prompt files
      };
      
      console.log('loadCustomFlowForCloning: Setting clonedFlow with hasVisualData:', hasVisualData);
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
      console.error('loadCustomFlowForCloning: Failed to load custom flow:', error);
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
      if (!response.ok) {
        throw new Error(`Failed to load flow YAML: ${response.status} ${response.statusText}`);
      }
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
      console.log('handleFlowCreated: Received flowData with visualNodes:', flowData.visualNodes?.length, 'visualEdges:', flowData.visualEdges?.length);
      
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
        visualNodes: flowData.visualNodes || [], // Save visual nodes for editor
        visualEdges: flowData.visualEdges || [], // Save visual edges for editor
        temp_flow_name: flowData.tempFlowName, // For prompt file copying (snake_case for backend)
        source_flow_name: flowData.sourceFlowName || clonedFlow?.sourceFlowName, // For copying prompts from source flow/template
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
    
    // If the editor was cleared, remove the draft from localStorage and reset
    if (draft?.cleared) {
      if (draftIdRef.current) {
        const existingDrafts = loadAllDrafts().filter(d => d.id !== draftIdRef.current);
        localStorage.setItem('wizard_drafts', JSON.stringify(existingDrafts));
      }
      draftIdRef.current = null;
      setDraftFlow(null);
      markStepValid(1, false);
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
    // If completed successfully, session state was already cleared and config was already saved
    // Skip autosave when we already completed a full Save or Save & Run
    // If user is cancelling (clicking Cancel button), clear session state
    // Note: We DON'T clear session state when user navigates away (refresh, clicking other nav)
    // because we want to restore the state when they come back
    
    // When editing and changes were made, save them (skip if already saved via Save/Save & Run)
    if (!completedSuccessfully && isEditMode && hasChanges && selectedFlow) {
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
    // For new flows (not editing), save as not_configured if partially filled (skip if already saved)
    else if (!completedSuccessfully && selectedFlow && (!modelConfig.model || !datasetConfig.data_files)) {
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
                Build a custom flow from scratch or modify existing flow templates
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
          {selectedFlow && (editingConfig || wizardData?.isCloning) ? (
            <Alert
              variant={AlertVariant.success}
              isInline
              title="Flow already selected"
              style={{ marginBottom: '20px', flexShrink: 0 }}
            >
              {editingConfig 
                ? <>Currently editing: <strong>{selectedFlow.name}</strong>. Click Next to proceed to configuration.</>
                : <>Cloning from: <strong>{selectedFlow.name}</strong>. You can select a different flow or click Next to proceed.</>
              }
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
              isImported={!!editingConfig || !!wizardData?.isCloning}
              onError={setErrorMessage}
              preSelectedFlowName={wizardData?.preSelectedFlowName || null}
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
          className="flow-builder-wizard-step"
          style={{ 
            height: '100%', 
            width: '100%',
            overflow: 'hidden',
            padding: 0
          }}
        >
          {/* Override PatternFly wizard step scrolling and style footer */}
          <style>{`
            .flow-builder-wizard-step {
              /* This class is on our wrapper */
            }
            /* Override the wizard main body scroll for this step only */
            .pf-v5-c-wizard__main-body:has(.flow-builder-wizard-step) {
              overflow: hidden !important;
            }
            /* Also handle older PatternFly versions */
            .pf-c-wizard__main-body:has(.flow-builder-wizard-step) {
              overflow: hidden !important;
            }
            /* Style the wizard footer to have clear separation from canvas */
            .pf-v5-c-wizard__footer,
            .pf-c-wizard__footer {
              border-top: 1px solid #d2d2d2;
              box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.08);
              background-color: #fff;
              position: relative;
              z-index: 10;
            }
          `}</style>
          <FlowBuilderPage
            key={`builder-${sourceType}-${selectedCloneFlow || selectedDraftId || editingConfig?.id || createdFlow?.metadata?.name || 'blank'}`}
            initialFlow={
              // Priority: 1) Already created/saved flow, 2) Cloned flow, 3) Draft flow, 4) null for blank
              createdFlow ? createdFlow :
              clonedFlow ? clonedFlow :
              sourceType === 'draft' ? draftFlow :
              null
            }
            isBlankStart={!createdFlow && !clonedFlow && sourceType !== 'draft'}
            onBack={() => {
              // User clicked back in FlowBuilderPage - reset all flow state at once
              resetFlowState();
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
    
    // Step 4: Dataset Source Selection
    // Shows source cards (upload preprocessed, upload & preprocess PDF, reuse previous)
    // This replaces the source selection that was inside DatasetConfigurationStep
    {
      id: 'dataset-source',
      name: 'Choose Dataset',
      component: (
        <div style={{ padding: '1.5rem 2.5rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Alert
            variant={AlertVariant.info}
            isInline
            title="Dataset source"
            style={{ marginBottom: '20px', flexShrink: 0 }}
          >
            Select how you want to provide data for your flow.
          </Alert>
          <div style={{ flex: 1, minHeight: 0 }}>
            <DatasetConfigurationStep
              datasetConfig={datasetConfig}
              importedConfig={null}
              onConfigChange={(config) => {
                setDatasetConfig(config);
                markStepValid(3, config.data_files ? true : false);
              }}
              onError={setErrorMessage}
              datasetSourceType={datasetSourceType}
              onDatasetSourceChange={(source) => {
                setDatasetSourceType(source);
                // Mark source step valid when source is chosen
                markStepValid(3, true);
              }}
              pdfPreprocessingState={pdfPreprocessingState}
              onPdfPreprocessingStateChange={(state) => setPdfPreprocessingState(state)}
              pdfDatasetInfo={pdfDatasetInfo}
              onPdfDatasetCreated={(info) => setPdfDatasetInfo(info)}
              // Force source selection mode
              forceSourceSelection={true}
            />
          </div>
        </div>
      ),
      enableNext: datasetSourceType !== 'none',
      canJumpTo: stepValidation[1] && stepValidation[2],
    },
    
    // Step 5 (conditional): PDF Preprocessing - only shown when user chose 'pdf'
    ...(datasetSourceType === 'pdf' ? [{
      id: 'pdf-preprocessing',
      name: 'Preprocess Data',
      component: (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '1rem 1.5rem' }}>
            <PDFPreprocessingStep
              requiredColumns={selectedFlow?.dataset_requirements?.required_columns || []}
              onDatasetCreated={(info) => {
                setPdfDatasetInfo(info);
              }}
              onCancel={null}
              onError={setErrorMessage}
              savedState={pdfPreprocessingState}
              onStateChange={(state) => setPdfPreprocessingState(state)}
            />
          </div>
        </div>
      ),
      enableNext: !!pdfDatasetInfo,
      canJumpTo: datasetSourceType === 'pdf',
    }] : []),

    // Step 5/6: Configure Dataset (loads the file, preview, sample config)
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
              datasetConfig={datasetConfig}
              importedConfig={null}
              onConfigChange={(config) => {
                setDatasetConfig(config);
                markStepValid(4, config.data_files ? true : false);
              }}
              onError={setErrorMessage}
              datasetSourceType={datasetSourceType}
              onDatasetSourceChange={(source) => setDatasetSourceType(source)}
              pdfPreprocessingState={pdfPreprocessingState}
              onPdfPreprocessingStateChange={(state) => setPdfPreprocessingState(state)}
              pdfDatasetInfo={pdfDatasetInfo}
              onPdfDatasetCreated={(info) => setPdfDatasetInfo(info)}
              // Skip source selection since it was done in step 4
              skipSourceSelection={true}
            />
          </div>
        </div>
      ),
      enableNext: stepValidation[4],
      canJumpTo: stepValidation[1] && stepValidation[2] && (datasetSourceType !== 'none'),
    },
    
    // Step 6/7: Dry Run
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
            onDryRunStateChange={setIsDryRunActive}
          />
        </div>
      ),
      enableNext: true,
      canJumpTo: stepValidation[1] && stepValidation[2] && stepValidation[4],
      isDryRunStep: true, // Flag to identify dry run step for navigation guard
    },
    
    // Review & Confirm
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
      canJumpTo: stepValidation[1] && stepValidation[2] && stepValidation[4],
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
    <ExecutionProvider value={executionContextValue}>
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
                <ReviewFooter
                  onSaveAndRun={handleSaveAndRun}
                  onSave={handleWizardSave}
                  isSaveAndRunning={isSaveAndRunning}
                />
              ) : step.customFooter ? (
                <BuildFlowFooter 
                  flowBuilderSaveInfo={flowBuilderSaveInfo} 
                  selectedFlow={selectedFlow}
                />
              ) : step.isDryRunStep ? (
                <DryRunFooter
                  isDryRunActive={isDryRunActive}
                  pendingNavigation={pendingNavigation}
                  onClearPendingNavigation={() => setPendingNavigation(null)}
                  onShowExitModal={(direction) => {
                    setPendingNavigation(direction);
                    setShowDryRunExitModal(true);
                  }}
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
      
      {/* Dry Run Exit Confirmation Modal */}
      <Modal
        variant={ModalVariant.small}
        title="Dry Run In Progress"
        isOpen={showDryRunExitModal}
        onClose={() => {
          setShowDryRunExitModal(false);
          setPendingNavigation(null);
        }}
        actions={[
          <Button
            key="stop-and-proceed"
            variant="danger"
            onClick={async () => {
              // Stop the dry run
              try {
                await fetch(`${API_BASE_URL}/api/flow/cancel-dry-run`, { method: 'POST' });
              } catch (error) {
                console.error('Error cancelling dry run:', error);
              }
              setShowDryRunExitModal(false);
              setIsDryRunActive(false);
              // Navigation is handled by DryRunFooter's useEffect which
              // watches isDryRunActive + pendingNavigation and triggers
              // goToNextStep/goToPrevStep when the dry run stops.
            }}
          >
            Stop Dry Run & Proceed
          </Button>,
          <Button
            key="cancel"
            variant="link"
            onClick={() => {
              setShowDryRunExitModal(false);
              setPendingNavigation(null);
            }}
          >
            Continue Dry Run
          </Button>
        ]}
      >
        <Alert
          variant={AlertVariant.warning}
          isInline
          title="A dry run is currently in progress"
          style={{ marginBottom: '16px' }}
        />
        <p>
          You are about to navigate away from the Dry Run step while a dry run is still running.
        </p>
        <p style={{ marginTop: '12px' }}>
          You can either:
        </p>
        <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
          <li><strong>Stop Dry Run & Proceed</strong> - Cancel the current dry run and navigate away</li>
          <li><strong>Continue Dry Run</strong> - Stay on this step and wait for the dry run to complete</li>
        </ul>
        <p style={{ marginTop: '12px', fontSize: '14px', color: '#6a6e73' }}>
          Note: You can view completed dry run results from the Flow Runs History page.
        </p>
      </Modal>
    </ExecutionProvider>
  );
};

export default UnifiedFlowWizard;

