import { useReducer, useCallback, useEffect, useMemo } from 'react';
import { preprocessingAPI, API_BASE_URL } from '../services/api';
import { useNotifications } from '../contexts/NotificationContext';

// ============================================================
// ACTION TYPES
// ============================================================
const SET_FIELD = 'SET_FIELD';
const UPDATE_FIELD = 'UPDATE_FIELD';
const MERGE_STATE = 'MERGE_STATE';
const RESET_STATE = 'RESET_STATE';
const RESTORE_STATE = 'RESTORE_STATE';

// ============================================================
// INITIAL STATE
// ============================================================
const DEFAULT_CUSTOM_ICL = {
  icl_document: '',
  icl_query_1: '',
  icl_response_1: '',
  icl_query_2: '',
  icl_response_2: '',
  icl_query_3: '',
  icl_response_3: '',
};

const initialState = {
  upload: {
    jobId: null,
    jobStatus: 'idle',
    uploadedFiles: [],
    isUploading: false,
    uploadError: null,
  },
  conversion: {
    progress: { current: 0, total: 0, message: '' },
    convertedFiles: [],
    error: null,
    selectedFiles: new Set(),
    filesBeingConverted: new Set(),
  },
  chunking: {
    chunkSize: 1000,
    chunkOverlap: 100,
    chunks: [],
    totalChunks: 0,
    isChunking: false,
    fileConfigs: {},
    selectedFiles: new Set(),
    previewFile: null,
    expandedPreviews: new Set(),
  },
  basicInfo: {
    domain: '',
    documentOutline: '',
    additionalColumns: {},
    fileInfo: {},
    editing: {},
  },
  modals: {
    chunkOpen: false,
    selectedChunk: null,
    comparisonOpen: false,
    comparisonFile: null,
    comparisonMarkdown: '',
    isLoadingMarkdown: false,
    datasetNameOpen: false,
    datasetName: '',
    pendingAction: null,
  },
  preview: {
    selectedColumns: {},
    selectedFiles: new Set(),
    samplesPerFile: 2,
    searchQuery: '',
    searchFocused: false,
    expandedFiles: new Set(),
    autoExpanded: false,
  },
  icl: {
    templates: [],
    selectedIndex: null,
    useCustom: false,
    custom: { ...DEFAULT_CUSTOM_ICL },
    fileConfigs: {},
    editingFile: {},
    expandedFiles: new Set(),
    allChunks: {},
    selectedChunkIdx: {},
    loadingChunks: {},
  },
  ui: {
    activeTab: 0,
    showChunksPreview: false,
    isCreating: false,
    expandedSteps: {
      step1: true,
      step2: true,
      step3: true,
      step4: true,
      step5: true,
      step6: true,
    },
  },
  datasetCreation: {
    snapshot: null,
    hasChanges: false,
    info: null,
  },
  stateRestored: false,
};

// ============================================================
// HELPERS
// ============================================================

/**
 * Immutably set a value at a dot-separated path in a nested object.
 */
function setNestedValue(obj, path, value) {
  const keys = path.split('.');
  if (keys.length === 1) {
    return { ...obj, [keys[0]]: value };
  }
  return {
    ...obj,
    [keys[0]]: setNestedValue(obj[keys[0]] || {}, keys.slice(1).join('.'), value),
  };
}

/**
 * Get a value at a dot-separated path from a nested object.
 */
function getNestedValue(obj, path) {
  return path.split('.').reduce((acc, key) => acc?.[key], obj);
}

// ============================================================
// REDUCER
// ============================================================
function pdfProcessingReducer(state, action) {
  switch (action.type) {
    case SET_FIELD:
      return setNestedValue(state, action.path, action.value);

    case UPDATE_FIELD: {
      const currentValue = getNestedValue(state, action.path);
      const newValue = action.updater(currentValue);
      return setNestedValue(state, action.path, newValue);
    }

    case MERGE_STATE:
      return {
        ...state,
        [action.slice]: {
          ...state[action.slice],
          ...action.values,
        },
      };

    case RESET_STATE:
      return {
        ...initialState,
        // Re-create Set instances to avoid shared references
        conversion: { ...initialState.conversion, selectedFiles: new Set(), filesBeingConverted: new Set() },
        chunking: { ...initialState.chunking, selectedFiles: new Set(), expandedPreviews: new Set() },
        preview: { ...initialState.preview, selectedFiles: new Set(), expandedFiles: new Set() },
        icl: { ...initialState.icl, custom: { ...DEFAULT_CUSTOM_ICL }, expandedFiles: new Set() },
        ui: { ...initialState.ui, expandedSteps: { ...initialState.ui.expandedSteps } },
      };

    case RESTORE_STATE: {
      const s = action.savedState;
      return {
        ...state,
        upload: {
          ...state.upload,
          jobId: s.jobId,
          jobStatus: s.jobStatus || 'idle',
          uploadedFiles: s.uploadedFiles || [],
        },
        conversion: {
          ...state.conversion,
          convertedFiles: s.convertedFiles || [],
        },
        chunking: {
          ...state.chunking,
          chunks: s.chunks || [],
          totalChunks: s.totalChunks || 0,
          chunkSize: s.chunkSize || 1000,
          chunkOverlap: s.chunkOverlap || 100,
          fileConfigs: s.fileChunkConfigs || {},
        },
        basicInfo: {
          ...state.basicInfo,
          domain: s.domain || '',
          documentOutline: s.documentOutline || '',
          additionalColumns: s.additionalColumns || {},
          fileInfo: s.fileBasicInfo || {},
        },
        icl: {
          ...state.icl,
          selectedIndex: s.selectedTemplateIndex ?? null,
          useCustom: s.useCustomICL || false,
          custom: s.customICL || { ...DEFAULT_CUSTOM_ICL },
          fileConfigs: s.fileICLConfigs || {},
        },
        datasetCreation: {
          ...state.datasetCreation,
          info: s.createdDatasetInfo || null,
        },
        stateRestored: true,
      };
    }

    default:
      return state;
  }
}

// ============================================================
// SETTER FACTORY
// ============================================================

/**
 * Creates a setter function compatible with React's useState setter API.
 * Supports both direct values and updater functions.
 */
function createSetter(dispatch, path) {
  return (valueOrUpdater) => {
    if (typeof valueOrUpdater === 'function') {
      dispatch({ type: UPDATE_FIELD, path, updater: valueOrUpdater });
    } else {
      dispatch({ type: SET_FIELD, path, value: valueOrUpdater });
    }
  };
}

// Session storage key prefix
const PDF_PREPROCESSING_STORAGE_KEY_PREFIX = 'sdg_hub_pdf_preprocessing_state_';

// ============================================================
// HOOK
// ============================================================

/**
 * Custom hook that manages all state for the PDF Preprocessing Step.
 * Uses useReducer internally with grouped state slices.
 *
 * Returns backward-compatible state values, setter functions,
 * handler functions, and computed values so the component JSX
 * requires minimal changes.
 */
export function usePDFProcessing({
  selectedFlow,
  requiredColumns = [],
  onDatasetCreated,
  onCancel,
  onError,
  savedState = null,
  onStateChange = null,
}) {
  const [state, dispatch] = useReducer(pdfProcessingReducer, initialState);

  // Notifications
  const { addSuccessNotification, addErrorNotification, addInfoNotification } = useNotifications();

  // ----------------------------------------------------------
  // Setters (created once since dispatch is stable)
  // ----------------------------------------------------------
  const setters = useMemo(() => {
    const s = (path) => createSetter(dispatch, path);
    return {
      // Upload
      setJobId: s('upload.jobId'),
      setJobStatus: s('upload.jobStatus'),
      setUploadedFiles: s('upload.uploadedFiles'),
      setIsUploading: s('upload.isUploading'),
      setUploadError: s('upload.uploadError'),
      // Conversion
      setConversionProgress: s('conversion.progress'),
      setConvertedFiles: s('conversion.convertedFiles'),
      setConversionError: s('conversion.error'),
      setSelectedFilesForConversion: s('conversion.selectedFiles'),
      setFilesBeingConverted: s('conversion.filesBeingConverted'),
      // Chunking
      setChunkSize: s('chunking.chunkSize'),
      setChunkOverlap: s('chunking.chunkOverlap'),
      setChunks: s('chunking.chunks'),
      setTotalChunks: s('chunking.totalChunks'),
      setIsChunking: s('chunking.isChunking'),
      setFileChunkConfigs: s('chunking.fileConfigs'),
      setSelectedFilesForChunking: s('chunking.selectedFiles'),
      setPreviewFile: s('chunking.previewFile'),
      setExpandedChunkPreviews: s('chunking.expandedPreviews'),
      // Basic Info
      setDomain: s('basicInfo.domain'),
      setDocumentOutline: s('basicInfo.documentOutline'),
      setAdditionalColumns: s('basicInfo.additionalColumns'),
      setFileBasicInfo: s('basicInfo.fileInfo'),
      setEditingBasicInfo: s('basicInfo.editing'),
      // Modals
      setChunkModalOpen: s('modals.chunkOpen'),
      setSelectedChunkForModal: s('modals.selectedChunk'),
      setComparisonModalOpen: s('modals.comparisonOpen'),
      setComparisonFile: s('modals.comparisonFile'),
      setComparisonMarkdownContent: s('modals.comparisonMarkdown'),
      setIsLoadingMarkdown: s('modals.isLoadingMarkdown'),
      setDatasetNameModalOpen: s('modals.datasetNameOpen'),
      setDatasetName: s('modals.datasetName'),
      setPendingAction: s('modals.pendingAction'),
      // Preview
      setPreviewSelectedColumns: s('preview.selectedColumns'),
      setPreviewSelectedFiles: s('preview.selectedFiles'),
      setPreviewSamplesPerFile: s('preview.samplesPerFile'),
      setPreviewFileSearchQuery: s('preview.searchQuery'),
      setPreviewFileSearchFocused: s('preview.searchFocused'),
      setPreviewExpandedFiles: s('preview.expandedFiles'),
      setPreviewAutoExpanded: s('preview.autoExpanded'),
      // ICL
      setIclTemplates: s('icl.templates'),
      setSelectedTemplateIndex: s('icl.selectedIndex'),
      setUseCustomICL: s('icl.useCustom'),
      setCustomICL: s('icl.custom'),
      setFileICLConfigs: s('icl.fileConfigs'),
      setEditingFileICL: s('icl.editingFile'),
      setExpandedICLFiles: s('icl.expandedFiles'),
      setFileAllChunks: s('icl.allChunks'),
      setSelectedChunkIdx: s('icl.selectedChunkIdx'),
      setLoadingChunks: s('icl.loadingChunks'),
      // UI
      setActiveTabKey: s('ui.activeTab'),
      setShowChunksPreview: s('ui.showChunksPreview'),
      setIsCreatingDataset: s('ui.isCreating'),
      setExpandedSteps: s('ui.expandedSteps'),
      // Dataset creation
      setCreatedDatasetInfo: s('datasetCreation.info'),
      setDatasetCreationSnapshot: s('datasetCreation.snapshot'),
      setHasChangesAfterCreation: s('datasetCreation.hasChanges'),
      // Top-level
      setStateRestored: s('stateRestored'),
    };
  }, [dispatch]);

  // ----------------------------------------------------------
  // Convenience aliases (read from nested state)
  // ----------------------------------------------------------
  const jobId = state.upload.jobId;
  const jobStatus = state.upload.jobStatus;
  const uploadedFiles = state.upload.uploadedFiles;
  const isUploading = state.upload.isUploading;
  const uploadError = state.upload.uploadError;

  const conversionProgress = state.conversion.progress;
  const convertedFiles = state.conversion.convertedFiles;
  const conversionError = state.conversion.error;
  const selectedFilesForConversion = state.conversion.selectedFiles;
  const filesBeingConverted = state.conversion.filesBeingConverted;

  const chunkSize = state.chunking.chunkSize;
  const chunkOverlap = state.chunking.chunkOverlap;
  const chunks = state.chunking.chunks;
  const totalChunks = state.chunking.totalChunks;
  const isChunking = state.chunking.isChunking;
  const fileChunkConfigs = state.chunking.fileConfigs;
  const selectedFilesForChunking = state.chunking.selectedFiles;
  const previewFile = state.chunking.previewFile;
  const expandedChunkPreviews = state.chunking.expandedPreviews;

  const domain = state.basicInfo.domain;
  const documentOutline = state.basicInfo.documentOutline;
  const additionalColumns = state.basicInfo.additionalColumns;
  const fileBasicInfo = state.basicInfo.fileInfo;
  const editingBasicInfo = state.basicInfo.editing;

  const chunkModalOpen = state.modals.chunkOpen;
  const selectedChunkForModal = state.modals.selectedChunk;
  const comparisonModalOpen = state.modals.comparisonOpen;
  const comparisonFile = state.modals.comparisonFile;
  const comparisonMarkdownContent = state.modals.comparisonMarkdown;
  const isLoadingMarkdown = state.modals.isLoadingMarkdown;
  const datasetNameModalOpen = state.modals.datasetNameOpen;
  const datasetName = state.modals.datasetName;
  const pendingAction = state.modals.pendingAction;

  const previewSelectedColumns = state.preview.selectedColumns;
  const previewSelectedFiles = state.preview.selectedFiles;
  const previewSamplesPerFile = state.preview.samplesPerFile;
  const previewFileSearchQuery = state.preview.searchQuery;
  const previewFileSearchFocused = state.preview.searchFocused;
  const previewExpandedFiles = state.preview.expandedFiles;
  const previewAutoExpanded = state.preview.autoExpanded;

  const iclTemplates = state.icl.templates;
  const selectedTemplateIndex = state.icl.selectedIndex;
  const useCustomICL = state.icl.useCustom;
  const customICL = state.icl.custom;
  const fileICLConfigs = state.icl.fileConfigs;
  const editingFileICL = state.icl.editingFile;
  const expandedICLFiles = state.icl.expandedFiles;
  const fileAllChunks = state.icl.allChunks;
  const selectedChunkIdx = state.icl.selectedChunkIdx;
  const loadingChunks = state.icl.loadingChunks;

  const activeTabKey = state.ui.activeTab;
  const showChunksPreview = state.ui.showChunksPreview;
  const isCreatingDataset = state.ui.isCreating;
  const expandedSteps = state.ui.expandedSteps;

  const createdDatasetInfo = state.datasetCreation.info;
  const datasetCreationSnapshot = state.datasetCreation.snapshot;
  const hasChangesAfterCreation = state.datasetCreation.hasChanges;

  const stateRestored = state.stateRestored;

  // ----------------------------------------------------------
  // Computed values
  // ----------------------------------------------------------
  const iclFields = requiredColumns.filter(col => col.startsWith('icl_'));
  const needsICL = iclFields.length > 0;
  const needsDomain = requiredColumns.includes('domain');
  const needsDocumentOutline = requiredColumns.includes('document_outline');
  const contentColumnName = requiredColumns.includes('text') ? 'text' : 'document';
  const needsBasicInfoStep = needsDomain || needsDocumentOutline;

  const storageKey = useMemo(() => {
    let flowName = 'default';
    if (typeof selectedFlow === 'string') {
      flowName = selectedFlow;
    } else if (selectedFlow && typeof selectedFlow === 'object' && selectedFlow.name) {
      flowName = selectedFlow.name;
    }
    const flowId = flowName.replace(/[^a-zA-Z0-9]/g, '_');
    return `${PDF_PREPROCESSING_STORAGE_KEY_PREFIX}${flowId}`;
  }, [selectedFlow]);

  // ----------------------------------------------------------
  // Effects
  // ----------------------------------------------------------

  // Save current state to session storage
  const saveStateToStorage = useCallback(() => {
    const stateToSave = {
      jobId,
      jobStatus,
      uploadedFiles,
      convertedFiles,
      chunks,
      totalChunks,
      chunkSize,
      chunkOverlap,
      domain,
      documentOutline,
      additionalColumns,
      selectedTemplateIndex,
      useCustomICL,
      customICL,
      createdDatasetInfo,
      fileChunkConfigs,
      fileBasicInfo,
      fileICLConfigs,
      timestamp: Date.now(),
    };
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(stateToSave));
      onStateChange?.(stateToSave);
    } catch (error) {
      console.warn('Failed to save PDF preprocessing state:', error);
    }
  }, [
    jobId, jobStatus, uploadedFiles, convertedFiles, chunks, totalChunks,
    chunkSize, chunkOverlap, domain, documentOutline, additionalColumns,
    selectedTemplateIndex, useCustomICL, customICL, createdDatasetInfo,
    fileChunkConfigs, fileBasicInfo, fileICLConfigs, storageKey, onStateChange,
  ]);

  // Restore state from parent-provided savedState
  useEffect(() => {
    if (stateRestored) return;
    try {
      sessionStorage.removeItem(storageKey);
    } catch (_e) {
      // Ignore storage errors
    }
    if (savedState && savedState.jobId) {
      dispatch({ type: RESTORE_STATE, savedState });
    } else {
      dispatch({ type: SET_FIELD, path: 'stateRestored', value: true });
    }
  }, [savedState, stateRestored, storageKey]);

  // Persist state on change
  useEffect(() => {
    if (stateRestored && jobId) {
      saveStateToStorage();
    }
  }, [
    stateRestored, jobId, jobStatus, uploadedFiles, convertedFiles,
    chunks, totalChunks, chunkSize, chunkOverlap, domain, documentOutline,
    additionalColumns, selectedTemplateIndex, useCustomICL, customICL,
    createdDatasetInfo, fileChunkConfigs, fileBasicInfo, fileICLConfigs,
    saveStateToStorage,
  ]);

  // Fetch ICL templates on mount
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await preprocessingAPI.getICLTemplates();
        dispatch({ type: SET_FIELD, path: 'icl.templates', value: response.templates || [] });
      } catch (error) {
        console.error('Failed to fetch ICL templates:', error);
      }
    };
    fetchTemplates();
  }, []);

  // Track changes after dataset creation
  useEffect(() => {
    if (!datasetCreationSnapshot || !createdDatasetInfo) return;
    const currentState = {
      fileChunkConfigs: JSON.stringify(fileChunkConfigs),
      fileBasicInfo: JSON.stringify(fileBasicInfo),
      uploadedFiles: JSON.stringify(uploadedFiles),
      domain,
      documentOutline,
      chunkSize,
      chunkOverlap,
    };
    const hasChanges =
      currentState.fileChunkConfigs !== datasetCreationSnapshot.fileChunkConfigs ||
      currentState.fileBasicInfo !== datasetCreationSnapshot.fileBasicInfo ||
      currentState.uploadedFiles !== datasetCreationSnapshot.uploadedFiles ||
      currentState.domain !== datasetCreationSnapshot.domain ||
      currentState.documentOutline !== datasetCreationSnapshot.documentOutline ||
      currentState.chunkSize !== datasetCreationSnapshot.chunkSize ||
      currentState.chunkOverlap !== datasetCreationSnapshot.chunkOverlap;
    dispatch({ type: SET_FIELD, path: 'datasetCreation.hasChanges', value: hasChanges });
  }, [
    fileChunkConfigs, fileBasicInfo, uploadedFiles, domain, documentOutline,
    chunkSize, chunkOverlap, datasetCreationSnapshot, createdDatasetInfo,
  ]);

  // ----------------------------------------------------------
  // Handler Functions
  // ----------------------------------------------------------

  /**
   * Handle file upload (PDF and MD files, supports adding to existing job)
   */
  const handleFileUpload = useCallback(async (event, addToExisting = false) => {
    const allFiles = Array.from(event.target.files);
    const validFiles = allFiles.filter(f => {
      const ext = f.name.toLowerCase();
      return ext.endsWith('.pdf') || ext.endsWith('.md');
    });

    if (validFiles.length === 0) {
      dispatch({ type: SET_FIELD, path: 'upload.uploadError', value: 'Please select PDF or Markdown files only.' });
      return;
    }

    dispatch({ type: MERGE_STATE, slice: 'upload', values: { isUploading: true, uploadError: null } });

    try {
      const existingJobId = addToExisting && jobId ? jobId : null;
      const response = await preprocessingAPI.uploadPDFs(validFiles, existingJobId);

      dispatch({ type: SET_FIELD, path: 'upload.jobId', value: response.job_id });
      dispatch({ type: SET_FIELD, path: 'upload.uploadedFiles', value: response.files });

      // Handle pre-converted MD files
      if (response.pre_converted_files && response.pre_converted_files.length > 0) {
        dispatch({
          type: UPDATE_FIELD,
          path: 'conversion.convertedFiles',
          updater: (prev) => {
            const existingOriginals = new Set(prev.map(cf => cf.original));
            const newConverted = response.pre_converted_files
              .filter(f => !existingOriginals.has(f.original))
              .map(f => ({
                original: f.original,
                markdown: f.markdown,
                path: f.path,
                isMarkdownUpload: true,
              }));
            return [...prev, ...newConverted];
          },
        });
      }

      // Auto-select new PDF files for conversion
      const allFileNames = response.files.map(f => f.name || f.filename);
      const alreadyConvertedNames = new Set([
        ...convertedFiles.map(cf => cf.original),
        ...(response.pre_converted_files || []).map(f => f.original),
      ]);
      const newPdfFileNames = allFileNames.filter(name =>
        !alreadyConvertedNames.has(name) && name.toLowerCase().endsWith('.pdf')
      );

      dispatch({
        type: UPDATE_FIELD,
        path: 'conversion.selectedFiles',
        updater: (prev) => {
          const newSet = new Set(prev);
          newPdfFileNames.forEach(name => newSet.add(name));
          return newSet;
        },
      });

      dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'uploaded' });
    } catch (error) {
      dispatch({
        type: SET_FIELD,
        path: 'upload.uploadError',
        value: error.response?.data?.detail || error.message || 'Failed to upload files',
      });
    } finally {
      dispatch({ type: SET_FIELD, path: 'upload.isUploading', value: false });
    }
  }, [jobId, convertedFiles]);

  /**
   * Start PDF to Markdown conversion
   */
  const handleConvert = useCallback(async (filesToConvert = null) => {
    if (!jobId) return;

    const filesArray = filesToConvert || Array.from(selectedFilesForConversion);
    if (filesArray.length === 0) return;

    dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'converting' });
    dispatch({ type: SET_FIELD, path: 'conversion.progress', value: { current: 0, total: filesArray.length, message: 'Starting conversion...' } });
    dispatch({ type: SET_FIELD, path: 'conversion.error', value: null });
    dispatch({ type: SET_FIELD, path: 'conversion.filesBeingConverted', value: new Set(filesArray) });

    try {
      const queryParam = `?selected_files=${encodeURIComponent(filesArray.join(','))}`;
      const response = await fetch(`${API_BASE_URL}/api/preprocessing/convert/${jobId}${queryParam}`, {
        method: 'POST',
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

        for (const line of lines) {
          try {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'start':
                dispatch({ type: SET_FIELD, path: 'conversion.progress', value: { current: 0, total: data.total_files, message: data.message } });
                break;
              case 'progress':
                dispatch({ type: SET_FIELD, path: 'conversion.progress', value: { current: data.current, total: data.total, message: data.message } });
                break;
              case 'file_complete': {
                const newFile = {
                  original: data.original || data.file?.original || data.file,
                  markdown: data.file?.markdown || data.file?.path || data.file,
                  path: data.file?.path,
                };
                dispatch({
                  type: UPDATE_FIELD,
                  path: 'conversion.convertedFiles',
                  updater: (prev) => {
                    const filtered = prev.filter(cf => cf.original !== newFile.original);
                    return [...filtered, newFile];
                  },
                });
                dispatch({
                  type: UPDATE_FIELD,
                  path: 'conversion.filesBeingConverted',
                  updater: (prev) => {
                    const newSet = new Set(prev);
                    newSet.delete(newFile.original);
                    return newSet;
                  },
                });
                dispatch({
                  type: UPDATE_FIELD,
                  path: 'conversion.progress',
                  updater: (prev) => ({
                    ...prev,
                    current: prev.current + 1,
                    message: `Converted ${newFile.original}`,
                  }),
                });
                break;
              }
              case 'file_error':
                console.warn('File conversion error:', data.message);
                addErrorNotification(`Failed to convert: ${data.message}`);
                break;
              case 'complete': {
                const numConverted = data.markdown_files?.length || filesArray.length;
                const totalConverted = data.total_converted || convertedFiles.length;

                dispatch({ type: SET_FIELD, path: 'conversion.selectedFiles', value: new Set() });
                dispatch({ type: SET_FIELD, path: 'conversion.filesBeingConverted', value: new Set() });
                dispatch({
                  type: UPDATE_FIELD,
                  path: 'conversion.progress',
                  updater: (prev) => ({ ...prev, message: data.message }),
                });

                setTimeout(() => {
                  if (totalConverted >= uploadedFiles.length) {
                    dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'converted' });
                  } else {
                    dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'uploaded' });
                  }
                }, 100);

                addSuccessNotification(
                  'PDF Conversion Complete!',
                  `Successfully converted ${numConverted} file(s) to Markdown.`
                );
                break;
              }
              case 'error':
                dispatch({ type: SET_FIELD, path: 'conversion.error', value: data.message });
                dispatch({ type: SET_FIELD, path: 'conversion.filesBeingConverted', value: new Set() });
                if (Object.values(fileChunkConfigs).some(c => c.isChunked)) {
                  dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'chunked' });
                } else if (convertedFiles.length > 0) {
                  dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'converted' });
                } else {
                  dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'uploaded' });
                }
                addErrorNotification('Conversion Error', data.message);
                break;
              default:
                break;
            }
          } catch (_e) {
            // Ignore parse errors for incomplete chunks
          }
        }
      }
    } catch (error) {
      dispatch({ type: SET_FIELD, path: 'conversion.error', value: error.message || 'Conversion failed' });
      dispatch({ type: SET_FIELD, path: 'conversion.filesBeingConverted', value: new Set() });
      dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'uploaded' });
      addErrorNotification('Conversion Failed', error.message || 'Unknown error occurred');
    }
  }, [jobId, selectedFilesForConversion, uploadedFiles.length, convertedFiles.length, fileChunkConfigs, addSuccessNotification, addErrorNotification]);

  /**
   * Handle document chunking
   */
  const handleChunk = useCallback(async (filesToChunk = null, customChunkSize = null, customChunkOverlap = null) => {
    if (!jobId) return;

    dispatch({ type: SET_FIELD, path: 'chunking.isChunking', value: true });

    const targetFiles = filesToChunk || Array.from(selectedFilesForChunking);
    const size = customChunkSize ?? chunkSize;
    const overlap = customChunkOverlap ?? chunkOverlap;

    try {
      const perFileConfigs = {};
      targetFiles.forEach(fileName => {
        perFileConfigs[fileName] = { chunk_size: size, overlap: overlap };
      });

      const response = await preprocessingAPI.chunkDocuments(jobId, {
        chunk_size: size,
        overlap: overlap,
        method: 'word',
        selected_files: targetFiles.length > 0 ? targetFiles : undefined,
        file_configs: Object.keys(perFileConfigs).length > 0 ? perFileConfigs : undefined,
      });

      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.fileConfigs',
        updater: (prev) => {
          const updated = { ...prev };
          targetFiles.forEach(fileName => {
            const fileChunks = (response.chunks_preview || []).filter(c =>
              c.source_file === fileName || c.source_file?.includes(fileName.replace('.pdf', ''))
            );
            const actualChunkCount = response.per_file_chunk_counts?.[fileName] || fileChunks.length;
            updated[fileName] = {
              chunkSize: size,
              chunkOverlap: overlap,
              isChunked: true,
              isConfirmed: false,
              needsReApply: false,
              chunks: fileChunks,
              totalChunks: actualChunkCount,
            };
          });
          return updated;
        },
      });

      dispatch({ type: SET_FIELD, path: 'chunking.totalChunks', value: response.total_chunks });
      dispatch({ type: SET_FIELD, path: 'chunking.chunks', value: response.chunks_preview || [] });

      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.expandedPreviews',
        updater: (prev) => {
          const newSet = new Set(prev);
          targetFiles.forEach(fileName => newSet.add(fileName));
          return newSet;
        },
      });

      dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'chunked' });
    } catch (error) {
      onError?.('Failed to chunk documents: ' + (error.response?.data?.detail || error.message));
    } finally {
      dispatch({ type: SET_FIELD, path: 'chunking.isChunking', value: false });
    }
  }, [jobId, chunkSize, chunkOverlap, selectedFilesForChunking, onError]);

  /**
   * Load a specific chunk by index for a file
   */
  const loadChunkForFile = useCallback(async (fileName, chunkIndex) => {
    dispatch({
      type: UPDATE_FIELD,
      path: 'icl.selectedChunkIdx',
      updater: (prev) => ({ ...prev, [fileName]: chunkIndex }),
    });

    let allChunks = fileAllChunks[fileName];

    if (!allChunks) {
      if (!jobId) return;

      dispatch({
        type: UPDATE_FIELD,
        path: 'icl.loadingChunks',
        updater: (prev) => ({ ...prev, [fileName]: true }),
      });
      try {
        const totalCount = fileChunkConfigs[fileName]?.totalChunks || 100;
        const response = await preprocessingAPI.getChunks(jobId, 0, Math.max(totalCount * 2, 500));
        const jobChunks = response.chunks || [];
        allChunks = jobChunks.filter(c =>
          c.source_file === fileName || c.source_file?.includes(fileName.replace('.pdf', ''))
        );
        dispatch({
          type: UPDATE_FIELD,
          path: 'icl.allChunks',
          updater: (prev) => ({ ...prev, [fileName]: allChunks }),
        });
      } catch (err) {
        console.error('Failed to fetch chunks:', err);
        allChunks = fileChunkConfigs[fileName]?.chunks || [];
      } finally {
        dispatch({
          type: UPDATE_FIELD,
          path: 'icl.loadingChunks',
          updater: (prev) => ({ ...prev, [fileName]: false }),
        });
      }
    }

    const chunkItem = allChunks[chunkIndex];
    if (chunkItem) {
      const chunkText = chunkItem.document || chunkItem.text || '';
      dispatch({
        type: UPDATE_FIELD,
        path: 'icl.editingFile',
        updater: (prev) => ({
          ...prev,
          [fileName]: { ...prev[fileName], useCustom: true, icl_document: chunkText },
        }),
      });
    }
  }, [jobId, fileAllChunks, fileChunkConfigs]);

  /**
   * Create the final dataset
   */
  const handleCreateDataset = useCallback(async () => {
    if (!jobId) return;

    dispatch({ type: SET_FIELD, path: 'ui.isCreating', value: true });

    try {
      let iclTemplate = null;
      if (needsICL) {
        if (useCustomICL) {
          iclTemplate = customICL;
        } else if (selectedTemplateIndex !== null && iclTemplates[selectedTemplateIndex]) {
          iclTemplate = iclTemplates[selectedTemplateIndex].template;
        }
      }

      const response = await preprocessingAPI.createDataset(jobId, {
        chunk_config: { chunk_size: chunkSize, overlap: chunkOverlap, method: 'word' },
        additional_columns: additionalColumns,
        icl_template: iclTemplate,
        domain: domain,
        document_outline: documentOutline,
        dataset_name: datasetName,
        content_column_name: contentColumnName,
        include_domain: needsDomain,
        include_document_outline: needsDocumentOutline,
      });

      dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'complete' });

      const datasetInfo = {
        file_path: response.file_path,
        num_records: response.num_records,
        columns: response.columns,
      };
      dispatch({ type: SET_FIELD, path: 'datasetCreation.info', value: datasetInfo });

      if (onDatasetCreated) {
        onDatasetCreated(datasetInfo);
      }

      dispatch({
        type: SET_FIELD,
        path: 'datasetCreation.snapshot',
        value: {
          fileChunkConfigs: JSON.stringify(fileChunkConfigs),
          fileBasicInfo: JSON.stringify(fileBasicInfo),
          uploadedFiles: JSON.stringify(uploadedFiles),
          domain,
          documentOutline,
          chunkSize,
          chunkOverlap,
        },
      });
      dispatch({ type: SET_FIELD, path: 'datasetCreation.hasChanges', value: false });

      addSuccessNotification('Dataset Created!', `Successfully created dataset with ${response.num_records} records.`);
    } catch (error) {
      onError?.('Failed to create dataset: ' + (error.response?.data?.detail || error.message));
    } finally {
      dispatch({ type: SET_FIELD, path: 'ui.isCreating', value: false });
    }
  }, [
    jobId, needsICL, useCustomICL, customICL, selectedTemplateIndex, iclTemplates,
    additionalColumns, domain, documentOutline, chunkSize, chunkOverlap,
    fileChunkConfigs, fileBasicInfo, uploadedFiles, datasetName,
    contentColumnName, needsDomain, needsDocumentOutline,
    addSuccessNotification, onError, onDatasetCreated,
  ]);

  /**
   * Download the created dataset
   */
  const handleDownloadDataset = useCallback(async (customName = null) => {
    if (!createdDatasetInfo?.file_path) return;

    try {
      const downloadUrl = `${API_BASE_URL}/${createdDatasetInfo.file_path}`;
      const fileName = customName || datasetName || createdDatasetInfo.file_path.split('/').pop() || 'preprocessed_dataset';
      const finalFileName = fileName.endsWith('.jsonl') ? fileName : `${fileName}.jsonl`;

      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = finalFileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      addSuccessNotification('Download Started', 'Your dataset file is being downloaded.');
    } catch (error) {
      addErrorNotification('Download Failed', error.message || 'Could not download the dataset.');
    }
  }, [createdDatasetInfo, datasetName, addSuccessNotification, addErrorNotification]);

  /**
   * Reset and start over
   */
  const handleReset = useCallback(async () => {
    if (jobId) {
      try {
        await preprocessingAPI.cleanup(jobId);
      } catch (_e) {
        // Ignore cleanup errors
      }
    }
    try {
      sessionStorage.removeItem(storageKey);
    } catch (_e) {
      // Ignore storage errors
    }
    dispatch({ type: RESET_STATE });
  }, [jobId, storageKey]);

  /**
   * Open comparison modal
   */
  const handleOpenComparison = useCallback(async (pdfFilename, markdownFilename) => {
    dispatch({ type: SET_FIELD, path: 'modals.comparisonFile', value: { pdfFilename, markdownFilename } });
    dispatch({ type: SET_FIELD, path: 'modals.comparisonOpen', value: true });
    dispatch({ type: SET_FIELD, path: 'modals.isLoadingMarkdown', value: true });
    dispatch({ type: SET_FIELD, path: 'modals.comparisonMarkdown', value: '' });

    try {
      const result = await preprocessingAPI.getMarkdownContent(jobId, markdownFilename);
      dispatch({ type: SET_FIELD, path: 'modals.comparisonMarkdown', value: result.content || '' });
    } catch (error) {
      console.error('Failed to load markdown content:', error);
      dispatch({ type: SET_FIELD, path: 'modals.comparisonMarkdown', value: 'Error loading markdown content' });
    } finally {
      dispatch({ type: SET_FIELD, path: 'modals.isLoadingMarkdown', value: false });
    }
  }, [jobId]);

  /**
   * Delete a single uploaded file
   */
  const handleDeleteFile = useCallback((fileToDelete) => {
    try {
      const fileName = fileToDelete.name || fileToDelete.filename;

      const updatedFiles = uploadedFiles.filter(f => {
        const currentFileName = f.name || f.filename;
        return currentFileName !== fileName;
      });
      dispatch({ type: SET_FIELD, path: 'upload.uploadedFiles', value: updatedFiles });

      // Remove from conversion state
      dispatch({
        type: UPDATE_FIELD,
        path: 'conversion.selectedFiles',
        updater: (prev) => { const ns = new Set(prev); ns.delete(fileName); return ns; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'conversion.filesBeingConverted',
        updater: (prev) => { const ns = new Set(prev); ns.delete(fileName); return ns; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'conversion.convertedFiles',
        updater: (prev) => prev.filter(cf => {
          const originalName = typeof cf === 'string' ? cf : cf.original;
          return originalName !== fileName && !originalName?.includes(fileName.replace('.pdf', ''));
        }),
      });

      // Remove from chunking state
      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.selectedFiles',
        updater: (prev) => { const ns = new Set(prev); ns.delete(fileName); return ns; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.expandedPreviews',
        updater: (prev) => { const ns = new Set(prev); ns.delete(fileName); return ns; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.fileConfigs',
        updater: (prev) => { const nc = { ...prev }; delete nc[fileName]; return nc; },
      });

      // Remove from basic info state
      dispatch({
        type: UPDATE_FIELD,
        path: 'basicInfo.fileInfo',
        updater: (prev) => { const ni = { ...prev }; delete ni[fileName]; return ni; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'basicInfo.editing',
        updater: (prev) => { const ni = { ...prev }; delete ni[fileName]; return ni; },
      });

      // Remove from ICL state
      dispatch({
        type: UPDATE_FIELD,
        path: 'icl.fileConfigs',
        updater: (prev) => { const nc = { ...prev }; delete nc[fileName]; return nc; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'icl.editingFile',
        updater: (prev) => { const ni = { ...prev }; delete ni[fileName]; return ni; },
      });
      dispatch({
        type: UPDATE_FIELD,
        path: 'icl.expandedFiles',
        updater: (prev) => { const ns = new Set(prev); ns.delete(fileName); return ns; },
      });

      // Recalculate chunks
      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.chunks',
        updater: (prev) => prev.filter(ch =>
          ch.source_file !== fileName && !ch.source_file?.includes(fileName.replace('.pdf', ''))
        ),
      });
      const deletedFileChunks = fileChunkConfigs[fileName]?.totalChunks || 0;
      dispatch({
        type: UPDATE_FIELD,
        path: 'chunking.totalChunks',
        updater: (prev) => Math.max(0, prev - deletedFileChunks),
      });

      // Reset if last file
      if (updatedFiles.length === 0) {
        dispatch({ type: SET_FIELD, path: 'upload.jobStatus', value: 'idle' });
        dispatch({ type: SET_FIELD, path: 'upload.jobId', value: null });
        dispatch({ type: SET_FIELD, path: 'conversion.convertedFiles', value: [] });
        dispatch({ type: SET_FIELD, path: 'chunking.chunks', value: [] });
        dispatch({ type: SET_FIELD, path: 'chunking.totalChunks', value: 0 });
        dispatch({ type: SET_FIELD, path: 'chunking.fileConfigs', value: {} });
        dispatch({ type: SET_FIELD, path: 'basicInfo.fileInfo', value: {} });
        dispatch({ type: SET_FIELD, path: 'basicInfo.editing', value: {} });
      }

      addInfoNotification('File Removed', `${fileName} has been removed from all steps`);
    } catch (error) {
      console.error('Error deleting file:', error);
      addErrorNotification('Delete Failed', `Could not remove ${fileToDelete.name || fileToDelete.filename}`);
    }
  }, [uploadedFiles, fileChunkConfigs, addInfoNotification, addErrorNotification]);

  /**
   * Check if we can proceed to create dataset
   */
  const canCreateDataset = useCallback(() => {
    if (totalChunks === 0) return false;

    if (needsBasicInfoStep) {
      if (!Object.values(fileBasicInfo).some(info => info.isComplete)) return false;
    }

    const requiredNonICL = requiredColumns.filter(col =>
      !col.startsWith('icl_') && col !== 'document' && col !== 'text' && col !== 'document_outline' && col !== 'domain'
    );

    for (const col of requiredNonICL) {
      if (!additionalColumns[col]) return false;
    }

    if (needsICL) {
      if (!Object.values(fileICLConfigs).some(config => config.isComplete)) {
        return false;
      }
    }

    return true;
  }, [fileBasicInfo, totalChunks, requiredColumns, additionalColumns, needsICL, fileICLConfigs, needsBasicInfoStep]);

  /**
   * Download a markdown file for a converted document
   */
  const handleDownloadMarkdown = useCallback((markdownFilename) => {
    if (markdownFilename && jobId) {
      const downloadUrl = `${API_BASE_URL}/api/preprocessing/download/${jobId}/${encodeURIComponent(markdownFilename)}`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = markdownFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      addSuccessNotification('Download Started', `Downloading ${markdownFilename}`);
    }
  }, [jobId, addSuccessNotification]);

  // ----------------------------------------------------------
  // Return
  // ----------------------------------------------------------
  return {
    // Raw state & dispatch for advanced patterns
    state,
    dispatch,

    // Convenience dispatch helpers
    setField: (path, value) => dispatch({ type: SET_FIELD, path, value }),
    mergeState: (slice, values) => dispatch({ type: MERGE_STATE, slice, values }),
    resetState: () => dispatch({ type: RESET_STATE }),

    // All individual state values
    jobId,
    jobStatus,
    uploadedFiles,
    isUploading,
    uploadError,
    conversionProgress,
    convertedFiles,
    conversionError,
    selectedFilesForConversion,
    filesBeingConverted,
    chunkSize,
    chunkOverlap,
    chunks,
    totalChunks,
    isChunking,
    fileChunkConfigs,
    selectedFilesForChunking,
    previewFile,
    expandedChunkPreviews,
    domain,
    documentOutline,
    additionalColumns,
    fileBasicInfo,
    editingBasicInfo,
    chunkModalOpen,
    selectedChunkForModal,
    comparisonModalOpen,
    comparisonFile,
    comparisonMarkdownContent,
    isLoadingMarkdown,
    datasetNameModalOpen,
    datasetName,
    pendingAction,
    previewSelectedColumns,
    previewSelectedFiles,
    previewSamplesPerFile,
    previewFileSearchQuery,
    previewFileSearchFocused,
    previewExpandedFiles,
    previewAutoExpanded,
    iclTemplates,
    selectedTemplateIndex,
    useCustomICL,
    customICL,
    fileICLConfigs,
    editingFileICL,
    expandedICLFiles,
    fileAllChunks,
    selectedChunkIdx,
    loadingChunks,
    activeTabKey,
    showChunksPreview,
    isCreatingDataset,
    expandedSteps,
    createdDatasetInfo,
    datasetCreationSnapshot,
    hasChangesAfterCreation,
    stateRestored,

    // All setter functions
    ...setters,

    // Handler functions
    handleFileUpload,
    handleConvert,
    handleChunk,
    loadChunkForFile,
    handleCreateDataset,
    handleDownloadDataset,
    handleReset,
    handleOpenComparison,
    handleDeleteFile,
    handleDownloadMarkdown,
    canCreateDataset,

    // Computed values
    iclFields,
    needsICL,
    needsDomain,
    needsDocumentOutline,
    contentColumnName,
    needsBasicInfoStep,
    storageKey,
  };
}

export default usePDFProcessing;
