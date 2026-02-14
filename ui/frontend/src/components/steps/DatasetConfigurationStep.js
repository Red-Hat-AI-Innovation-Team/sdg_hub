import React, { useState, useReducer, useEffect, useCallback, useRef } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Form,
  FormGroup,
  TextInput,
  NumberInput,
  Checkbox,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  Grid,
  GridItem,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  ExpandableSection,
  CodeBlock,
  CodeBlockCode,
  List,
  ListItem,
  ToggleGroup,
  ToggleGroupItem,
  FileUpload,
  Modal,
  ModalVariant,
  Radio,
} from '@patternfly/react-core';
import { CheckCircleIcon, UploadIcon, EditIcon, ExclamationTriangleIcon, FileIcon, FilePdfIcon, OutlinedQuestionCircleIcon, HistoryIcon } from '@patternfly/react-icons';
import { Tooltip } from '@patternfly/react-core';
import { datasetAPI } from '../../services/api';
import PDFPreprocessingStep from './PDFPreprocessingStep';
import MissingColumnsModal from './MissingColumnsModal';
import DuplicatesModal from './DuplicatesModal';
import { useExecutionConfig } from '../../contexts/ExecutionContext';

// --- Reducer definitions ---

const uploadInitialState = {
  uploadedFile: null,
  uploadedFileName: '',
  uploadedFilePath: '',
  isUploadLoading: false,
};

function uploadReducer(state, action) {
  switch (action.type) {
    case 'SET_FILE': return { ...state, uploadedFile: action.payload };
    case 'SET_FILE_NAME': return { ...state, uploadedFileName: action.payload };
    case 'SET_FILE_PATH': return { ...state, uploadedFilePath: action.payload };
    case 'SET_LOADING': return { ...state, isUploadLoading: action.payload };
    case 'CLEAR_FILE': return { ...state, uploadedFile: null, uploadedFileName: '' };
    case 'CLEAR_ALL': return uploadInitialState;
    default: return state;
  }
}

const missingColsInitialState = {
  missingColumns: [],
  showMissingColumnsModal: false,
  addingMissingColumns: false,
  currentMissingColumnIndex: 0,
  missingColumnValues: {},
  currentColumnInput: '',
};

function missingColsReducer(state, action) {
  switch (action.type) {
    case 'SET_MISSING_COLUMNS': return { ...state, missingColumns: action.payload };
    case 'SHOW_MODAL': return { ...state, showMissingColumnsModal: true };
    case 'HIDE_MODAL': return { ...state, showMissingColumnsModal: false };
    case 'START_ADDING':
      return {
        ...state,
        showMissingColumnsModal: false,
        addingMissingColumns: true,
        currentMissingColumnIndex: 0,
        missingColumnValues: {},
        currentColumnInput: '',
      };
    case 'SET_ADDING': return { ...state, addingMissingColumns: action.payload };
    case 'SET_COLUMN_INDEX': return { ...state, currentMissingColumnIndex: action.payload };
    case 'SET_COLUMN_VALUES': return { ...state, missingColumnValues: action.payload };
    case 'SET_COLUMN_INPUT': return { ...state, currentColumnInput: action.payload };
    case 'SAVE_COLUMN_VALUE':
      return {
        ...state,
        missingColumnValues: { ...state.missingColumnValues, [action.columnName]: action.value },
        currentColumnInput: '',
      };
    case 'NEXT_COLUMN':
      return { ...state, currentMissingColumnIndex: state.currentMissingColumnIndex + 1 };
    case 'CANCEL_FIX_MANUALLY':
      return { ...state, showMissingColumnsModal: false, missingColumns: [] };
    case 'CANCEL_ADDING':
      return {
        ...state,
        addingMissingColumns: false,
        missingColumnValues: {},
        currentMissingColumnIndex: 0,
        currentColumnInput: '',
      };
    case 'RESET': return missingColsInitialState;
    default: return state;
  }
}

const duplicatesInitialState = {
  showDuplicatesModal: false,
  duplicateInfo: null,
  isRemovingDuplicates: false,
};

function duplicatesReducer(state, action) {
  switch (action.type) {
    case 'SHOW_MODAL': return { ...state, showDuplicatesModal: true, duplicateInfo: action.payload };
    case 'HIDE_MODAL': return { showDuplicatesModal: false, duplicateInfo: null, isRemovingDuplicates: false };
    case 'SET_REMOVING': return { ...state, isRemovingDuplicates: action.payload };
    default: return state;
  }
}

const unsupportedFormatInitialState = {
  showUnsupportedFormatError: false,
  unsupportedFileName: '',
};

function unsupportedFormatReducer(state, action) {
  switch (action.type) {
    case 'SHOW': return { showUnsupportedFormatError: true, unsupportedFileName: action.payload };
    case 'HIDE': return { ...state, showUnsupportedFormatError: false };
    default: return state;
  }
}

/**
 * Dataset Configuration Step Component
 * 
 * Allows users to:
 * - Choose between preprocessed dataset or PDF preprocessing
 * - View dataset schema requirements
 * - Load dataset from file
 * - Configure dataset parameters (num_samples, shuffle, seed)
 * - Preview loaded dataset
 */
const DatasetConfigurationStep = ({ 
  selectedFlow: selectedFlowProp, datasetConfig, importedConfig, onConfigChange, onError,
  // Props for wizard-level state management (lifted from internal state)
  datasetSourceType: externalDatasetSourceType,
  onDatasetSourceChange,
  pdfPreprocessingState: externalPdfState,
  onPdfPreprocessingStateChange,
  pdfDatasetInfo: externalPdfDatasetInfo,
  onPdfDatasetCreated,
  // Wizard step split props
  forceSourceSelection, // When true, only render source selection cards
  skipSourceSelection,  // When true, skip source selection and go directly to the appropriate mode
}) => {
  // Use ExecutionContext for selectedFlow (eliminates prop drilling), with prop fallback
  const { selectedFlow: selectedFlowFromContext } = useExecutionConfig();
  const selectedFlow = selectedFlowFromContext || selectedFlowProp;

  // Source selection state: 'none' | 'preprocessed' | 'pdf' | 'reuse'
  // Use external state if provided (wizard-level), otherwise use internal state
  const [internalDatasetSource, setInternalDatasetSource] = useState('none');
  const datasetSource = externalDatasetSourceType || internalDatasetSource;
  const setDatasetSource = (source) => {
    if (onDatasetSourceChange) {
      onDatasetSourceChange(source);
    }
    setInternalDatasetSource(source);
  };
  
  // PDF preprocessing state (for back navigation) - use external if provided
  const [internalPdfState, setInternalPdfState] = useState(null);
  const pdfPreprocessingState = externalPdfState !== undefined ? externalPdfState : internalPdfState;
  const setPdfPreprocessingState = (state) => {
    if (onPdfPreprocessingStateChange) {
      onPdfPreprocessingStateChange(state);
    }
    setInternalPdfState(state);
  };
  
  // Session storage key for PDF preprocessing state (flow-specific)
  const pdfStateStorageKey = `sdg_hub_wizard_pdf_state_${typeof selectedFlow === 'string' ? selectedFlow : (selectedFlow?.name || 'default')}`;

  // Restore PDF preprocessing state on mount (for same wizard session)
  useEffect(() => {
    try {
      const savedPdfState = sessionStorage.getItem(pdfStateStorageKey);
      if (savedPdfState) {
        const parsed = JSON.parse(savedPdfState);
        // Only restore if state is recent (within 1 hour) and has a jobId
        if (parsed.jobId && parsed.timestamp && (Date.now() - parsed.timestamp) < 60 * 60 * 1000) {
          setPdfPreprocessingState(parsed);
          // Auto-navigate to PDF preprocessing if user had work in progress
          setDatasetSource('pdf');
        } else {
          // Clear stale state
          sessionStorage.removeItem(pdfStateStorageKey);
        }
      }
    } catch (error) {
      console.warn('Failed to restore PDF preprocessing state:', error);
    }
  }, [pdfStateStorageKey]);

  // Save PDF preprocessing state to session storage when it changes
  useEffect(() => {
    if (pdfPreprocessingState && pdfPreprocessingState.jobId) {
      try {
        sessionStorage.setItem(pdfStateStorageKey, JSON.stringify(pdfPreprocessingState));
      } catch (error) {
        console.warn('Failed to save PDF preprocessing state:', error);
      }
    }
  }, [pdfPreprocessingState, pdfStateStorageKey]);

  // Clear PDF state from session storage when dataset is fully configured
  const clearPdfSessionState = useCallback(() => {
    try {
      sessionStorage.removeItem(pdfStateStorageKey);
      setPdfPreprocessingState(null);
    } catch (error) {
      // Ignore
    }
  }, [pdfStateStorageKey]);
  
  // Previously used datasets state
  const [previousDatasets, setPreviousDatasets] = useState([]);
  const [selectedPreviousDataset, setSelectedPreviousDataset] = useState(null);
  
  const [schema, setSchema] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigured, setIsConfigured] = useState(false);
  const [isFromReuse, setIsFromReuse] = useState(false); // Track if dataset was loaded via reuse selection

  // Form state
  const [dataFiles, setDataFiles] = useState('');
  const [numSamples, setNumSamples] = useState(2);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42);
  const [split, setSplit] = useState('train');
  
  // Flag to skip reload when we just configured the dataset internally.
  // Using a ref (not state) so that clearing it doesn't re-trigger the useEffect.
  const skipNextConfigReloadRef = useRef(false);

  // Pre-fill form with existing datasetConfig or imported configuration
  useEffect(() => {
    // Skip reload if flag is set (we just configured from within this component)
    if (skipNextConfigReloadRef.current) {
      skipNextConfigReloadRef.current = false;
      return;
    }
    
    const configToUse = importedConfig || datasetConfig;
    
    if (configToUse && Object.keys(configToUse).length > 0) {
      if (configToUse.data_files) {
        setDataFiles(configToUse.data_files);
        dispatchUpload({ type: 'SET_FILE_PATH', payload: configToUse.data_files }); // Set the file path for validation
        
        // Set uploaded filename from config
        if (configToUse.uploaded_file) {
          dispatchUpload({ type: 'SET_FILE_NAME', payload: configToUse.uploaded_file });
        } else if (configToUse.data_files) {
          // Extract filename from path if uploaded_file not provided
          const pathParts = configToUse.data_files.split('/');
          dispatchUpload({ type: 'SET_FILE_NAME', payload: pathParts[pathParts.length - 1] });
        }
      }
      if (configToUse.split) setSplit(configToUse.split);
      if (configToUse.num_samples) setNumSamples(configToUse.num_samples);
      if (configToUse.shuffle !== undefined) setShuffle(configToUse.shuffle);
      if (configToUse.seed) setSeed(configToUse.seed);
      setIsConfigured(true);
      
      // Auto-load preview for existing configurations
      if (configToUse.data_files && selectedFlow) {
        loadExistingDatasetPreview(configToUse);
      }
    }
  }, [importedConfig, datasetConfig]);

  // File upload state (useReducer)
  const [uploadState, dispatchUpload] = useReducer(uploadReducer, uploadInitialState);
  const { uploadedFile, uploadedFileName, uploadedFilePath, isUploadLoading } = uploadState;

  // UI state
  const [isPreviewExpanded, setIsPreviewExpanded] = useState(false);
  const [selectedColumnsPerSample, setSelectedColumnsPerSample] = useState({}); // Track selected column for each sample
  
  // Missing columns validation (useReducer)
  const [missingColsState, dispatchMissingCols] = useReducer(missingColsReducer, missingColsInitialState);
  const { missingColumns, showMissingColumnsModal, addingMissingColumns, currentMissingColumnIndex, missingColumnValues, currentColumnInput } = missingColsState;
  
  // Duplicate rows detection (useReducer)
  const [duplicatesState, dispatchDuplicates] = useReducer(duplicatesReducer, duplicatesInitialState);
  const { showDuplicatesModal, duplicateInfo, isRemovingDuplicates } = duplicatesState;

  /**
   * Load dataset schema when flow is selected
   */
  useEffect(() => {
    if (selectedFlow) {
      loadSchema();
    }
  }, [selectedFlow]);

  /**
   * Load previously used datasets from localStorage
   */
  useEffect(() => {
    try {
      const stored = localStorage.getItem('sdg_hub_previous_datasets');
      console.log('Loading previous datasets from localStorage:', stored);
      if (stored) {
        const datasets = JSON.parse(stored);
        console.log('Loaded previous datasets:', datasets);
        setPreviousDatasets(datasets);
      } else {
        console.log('No previous datasets found in localStorage');
      }
    } catch (error) {
      console.warn('Failed to load previous datasets:', error);
    }
  }, []);

  /**
   * Save dataset to previously used list when configuration is complete
   * @param {object} config - Dataset configuration
   * @param {string} source - 'preprocessed' or 'pdf'
   * @param {array} columns - Optional columns array (pass directly since state may not be updated yet)
   */
  const saveDatasetToHistory = (config, source, columns = null) => {
    try {
      console.log('Saving dataset to history:', { config, source, columns });
      
      const stored = localStorage.getItem('sdg_hub_previous_datasets') || '[]';
      const datasets = JSON.parse(stored);
      
      // Create entry
      const entry = {
        id: Date.now(),
        name: config.uploaded_file || config.data_files?.split('/').pop() || 'Unknown',
        data_files: config.data_files,
        source: source, // 'preprocessed' or 'pdf'
        flow: selectedFlow?.name || selectedFlow?.flow_id || 'Unknown flow',
        timestamp: new Date().toISOString(),
        num_samples: config.num_samples,
        columns: columns || preview?.columns || [],
      };
      
      console.log('Dataset entry to save:', entry);
      
      // Check if already exists (by data_files path)
      const existingIdx = datasets.findIndex(d => d.data_files === config.data_files);
      if (existingIdx >= 0) {
        // Update existing entry
        datasets[existingIdx] = { ...datasets[existingIdx], ...entry, id: datasets[existingIdx].id };
        console.log('Updated existing dataset entry');
      } else {
        // Add new entry (limit to last 20)
        datasets.unshift(entry);
        if (datasets.length > 20) {
          datasets.pop();
        }
        console.log('Added new dataset entry, total:', datasets.length);
      }
      
      localStorage.setItem('sdg_hub_previous_datasets', JSON.stringify(datasets));
      setPreviousDatasets(datasets);
      console.log('Dataset saved to localStorage successfully');
    } catch (error) {
      console.error('Failed to save dataset to history:', error);
    }
  };

  /**
   * Load dataset schema from API
   */
  const loadSchema = async () => {
    try {
      setLoading(true);
      
      // If selectedFlow has dataset_requirements, use those (for custom flows)
      if (selectedFlow?.dataset_requirements) {
        setSchema(selectedFlow.dataset_requirements);
      } else {
        // Otherwise try to load from backend API (for existing flows)
        try {
          const data = await datasetAPI.getSchema();
          setSchema(data);
        } catch (apiError) {
          // If API fails (e.g., custom flow with no backend flow), use empty schema
          console.warn('Could not load schema from backend, using defaults:', apiError.message);
          setSchema({
            required_columns: [],
            optional_columns: [],
            description: 'No schema requirements defined'
          });
        }
      }
    } catch (error) {
      console.error('Failed to load dataset schema:', error);
      // Use empty schema as fallback
      setSchema({
        required_columns: [],
        optional_columns: [],
        description: 'No schema requirements defined'
      });
    } finally {
      setLoading(false);
    }
  };

  /**
   * Load preview for existing dataset configuration (when editing)
   */
  const loadExistingDatasetPreview = async (config) => {
    try {
      
      // Load dataset to backend (so preview API can work)
      const loadConfig = {
        data_files: config.data_files,
        split: config.split || 'train',
        num_samples: config.num_samples || null,
        shuffle: config.shuffle !== undefined ? config.shuffle : true,
        seed: config.seed || 42,
        // Include added_columns if present in config
        added_columns: config.added_columns || null,
      };
      
      await datasetAPI.loadDataset(loadConfig);
      
      // Get preview
      const previewData = await datasetAPI.getPreview();
      setPreview(previewData);
    } catch (error) {
      console.warn('Could not auto-load preview for existing dataset:', error.message);
      // Don't show error to user - preview is optional
    }
  };

  // Supported file formats
  const SUPPORTED_FORMATS = ['jsonl', 'json', 'csv', 'parquet', 'pq'];
  
  // Unsupported format error state (useReducer)
  const [unsupportedFormatState, dispatchUnsupportedFormat] = useReducer(unsupportedFormatReducer, unsupportedFormatInitialState);
  const { showUnsupportedFormatError, unsupportedFileName } = unsupportedFormatState;

  /**
   * Get file format from filename extension
   */
  const getFileFormat = (filename) => {
    const ext = filename.toLowerCase().split('.').pop();
    const formatMap = {
      'jsonl': 'jsonl',
      'json': 'json',
      'csv': 'csv',
      'parquet': 'parquet',
      'pq': 'parquet'
    };
    return formatMap[ext] || null;  // Return null for unsupported formats
  };

  /**
   * Check if file format is supported
   */
  const isFormatSupported = (filename) => {
    const ext = filename.toLowerCase().split('.').pop();
    return SUPPORTED_FORMATS.includes(ext);
  };

  /**
   * Handle file upload - supports multiple formats (JSONL, JSON, CSV, Parquet)
   */
  const handleFileUpload = async (event, file) => {
    // Validate file format first
    if (!isFormatSupported(file.name)) {
      dispatchUnsupportedFormat({ type: 'SHOW', payload: file.name });
      return;  // Don't proceed with upload
    }
    
    dispatchUpload({ type: 'SET_LOADING', payload: true });
    try {
      const fileFormat = getFileFormat(file.name);
      const isBinaryFormat = fileFormat === 'parquet';
      
      // For binary files (Parquet), upload directly without reading content
      if (isBinaryFormat) {
        dispatchUpload({ type: 'SET_FILE', payload: file });  // Store file object for binary formats
        dispatchUpload({ type: 'SET_FILE_NAME', payload: file.name });
        
        // Upload directly and let backend handle it
        await validateUploadedFile(null, file, null, fileFormat);
        dispatchUpload({ type: 'SET_LOADING', payload: false });
        return;
      }
      
      // For text files (JSON, JSONL, CSV), read content
      const reader = new FileReader();
      reader.onload = async (e) => {
        const fileContent = e.target.result;
        dispatchUpload({ type: 'SET_FILE', payload: fileContent });
        dispatchUpload({ type: 'SET_FILE_NAME', payload: file.name });
        
        // Count samples based on format
        let sampleCount = numSamples;
        try {
          if (fileFormat === 'jsonl') {
            // JSONL: count non-empty lines
            const lines = fileContent.split('\n').filter(line => line.trim().length > 0);
            sampleCount = lines.length;
          } else if (fileFormat === 'csv') {
            // CSV: count lines minus header
            const lines = fileContent.split('\n').filter(line => line.trim().length > 0);
            sampleCount = Math.max(1, lines.length - 1);
          } else if (fileFormat === 'json') {
            // JSON: try to parse and count array length
            const parsed = JSON.parse(fileContent);
            sampleCount = Array.isArray(parsed) ? parsed.length : 1;
          }
          setNumSamples(sampleCount);
        } catch (error) {
          // Silently fail - keep default value
        }
        
        // Validate and upload
        await validateUploadedFile(fileContent, file, sampleCount, fileFormat);
        dispatchUpload({ type: 'SET_LOADING', payload: false });
      };
      reader.onerror = () => {
        onError('Failed to read file');
        dispatchUpload({ type: 'SET_LOADING', payload: false });
      };
      reader.readAsText(file);
    } catch (error) {
      onError('Error reading file: ' + error.message);
      dispatchUpload({ type: 'SET_LOADING', payload: false });
    }
  };

  /**
   * Handle file upload - just upload the file to backend and show preview
   * User must click "Load Dataset" button to finalize configuration
   * @param {string|null} fileContent - The file content (null for binary files)
   * @param {File} file - The file object
   * @param {number|null} actualSampleCount - The actual sample count (null for binary files)
   * @param {string} fileFormat - The detected file format (jsonl, json, csv, parquet, auto)
   */
  const validateUploadedFile = async (fileContent, file, actualSampleCount, fileFormat = 'auto') => {
    let uploadedPath = null;
    try {
      // Always use the original file object for upload
      // This ensures proper multipart form handling
      const fileObj = file;

      // Upload the file to the backend
      const uploadResponse = await datasetAPI.uploadFile(fileObj);
      uploadedPath = uploadResponse.file_path;
      dispatchUpload({ type: 'SET_FILE_PATH', payload: uploadedPath });
      setDataFiles(uploadedPath);
      
      // Optionally get a preview for user reference (but don't auto-configure)
      try {
        const loadConfig = {
          data_files: uploadedPath,
          file_format: fileFormat,  // Pass format for optimal loading
          num_samples: actualSampleCount || null,
          shuffle,
          seed,
        };

        await datasetAPI.loadDataset(loadConfig);
        
        // Get preview to show columns
        const previewData = await datasetAPI.getPreview();
        setPreview(previewData);
        
        // Update sample count from preview (especially for binary formats)
        if (previewData.num_samples && !actualSampleCount) {
          setNumSamples(previewData.num_samples);
        }
      } catch (previewError) {
        // Preview is optional - user can still load manually
        console.warn('Could not get preview during upload:', previewError);
      }
      
      // DON'T auto-configure - user must click "Load Dataset" button
      // Just mark that file is uploaded and ready to be configured
      
    } catch (error) {
      console.error('File upload error:', error);
      // Clear the uploaded file state since upload failed
      dispatchUpload({ type: 'CLEAR_ALL' });
      setDataFiles('');
      
      // Show error to user
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      alert(`Failed to upload file: ${errorMessage}`);
    }
  };

  /**
   * Clear uploaded file
   */
  const handleClearUpload = () => {
    dispatchUpload({ type: 'CLEAR_FILE' });
    // Reset number of samples to default when clearing
    setNumSamples(2);
  };

  /**
   * Check if dataset has all required columns
   */
  const checkMissingColumns = (previewData) => {
    // Get required columns from schema
    const requiredColumns = schema?.required_columns || schema?.requirements?.required_columns || [];
    
    if (requiredColumns.length === 0) {
      return []; // No requirements, all good
    }
    
    // Get columns from preview data
    const datasetColumns = previewData.columns || [];
    
    // Find missing columns
    const missing = requiredColumns.filter(col => !datasetColumns.includes(col));
    
    
    return missing;
  };

  /**
   * Handle dataset loading/reloading (when user clicks Load/Reload button)
   */
  const handleLoadUploadedDataset = async () => {
    try {
      setIsLoading(true);

      // Use the stored uploaded path (set during file upload)
      const uploadedPath = uploadedFilePath || dataFiles;
      
      if (!uploadedPath) {
        throw new Error('No file to load');
      }
      
      // Detect file format from filename
      const fileFormat = getFileFormat(uploadedFileName || uploadedPath);
      
      // Reload the dataset WITH the user's current parameters
      const loadConfig = {
        data_files: uploadedPath,
        file_format: fileFormat,  // Pass format for optimal pandas loading
        num_samples: numSamples || null,
        shuffle,
        seed,
      };

      // Load the filtered dataset with new parameters
      const response = await datasetAPI.loadDataset(loadConfig);
      
      // Get updated preview
      const previewData = await datasetAPI.getPreview();
      setPreview(previewData);

      // Check for duplicates FIRST (before missing columns check)
      try {
        const duplicatesResult = await datasetAPI.checkDuplicates();
        if (duplicatesResult.has_duplicates) {
          // Store duplicate info and show modal
          dispatchDuplicates({ type: 'SHOW_MODAL', payload: duplicatesResult });
          setIsLoading(false);
          return; // Don't continue - wait for user to handle duplicates
        }
      } catch (dupError) {
        // If duplicate check fails, continue anyway (non-critical)
        console.warn('Could not check for duplicates:', dupError);
      }

      // Check for missing columns
      const missing = checkMissingColumns(previewData);
      
      if (missing.length > 0) {
        // Dataset is missing required columns - show modal
        dispatchMissingCols({ type: 'SET_MISSING_COLUMNS', payload: missing });
        dispatchMissingCols({ type: 'SHOW_MODAL' });
        setIsLoading(false);
        return; // Don't configure yet - wait for user to add columns
      }

      // All columns present - configure
      const finalConfig = {
        data_files: uploadedPath,
        file_format: fileFormat,
        num_samples: numSamples || null,
        shuffle,
        seed,
        uploaded_file: uploadedFileName
      };

      // Skip the next config reload since we just configured the dataset
      skipNextConfigReloadRef.current = true;
      
      // Update parent state
      onConfigChange(finalConfig);
      setIsConfigured(true);
      setIsFromReuse(false); // Clear reuse flag after explicit load
      
      // Save to history (pass columns directly since state may not be updated yet)
      saveDatasetToHistory(finalConfig, 'preprocessed', previewData?.columns || []);

    } catch (error) {
      onError('Failed to reload dataset: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle removing duplicates and continuing with dataset loading
   */
  const handleRemoveDuplicates = async () => {
    try {
      dispatchDuplicates({ type: 'SET_REMOVING', payload: true });
      
      // Remove duplicates from the loaded dataset
      const result = await datasetAPI.removeDuplicates();
      
      // Close modal
      dispatchDuplicates({ type: 'HIDE_MODAL' });
      
      // Update preview with new data
      const previewData = await datasetAPI.getPreview();
      setPreview(previewData);
      
      // Update numSamples to reflect the new count
      if (result.new_count) {
        setNumSamples(result.new_count);
      }
      
      // Now continue with the rest of the loading process (check missing columns)
      const missing = checkMissingColumns(previewData);
      
      if (missing.length > 0) {
        // Dataset is missing required columns - show modal
        dispatchMissingCols({ type: 'SET_MISSING_COLUMNS', payload: missing });
        dispatchMissingCols({ type: 'SHOW_MODAL' });
        return;
      }

      // All columns present - configure.
      // Use the deduplicated file path returned by the backend so the
      // generation worker (which reloads from disk) gets the clean data.
      const dedupPath = result.dedup_data_files;
      const uploadedPath = dedupPath || uploadedFilePath || dataFiles;
      const fileFormat = dedupPath ? 'jsonl' : getFileFormat(uploadedFileName || uploadedPath);
      
      const finalConfig = {
        data_files: uploadedPath,
        file_format: fileFormat,
        // The dedup file already contains the exact rows we want, so
        // clear num_samples and shuffle to avoid re-processing.
        num_samples: dedupPath ? null : (result.new_count || numSamples || null),
        shuffle: dedupPath ? false : shuffle,
        seed,
        uploaded_file: uploadedFileName
      };

      skipNextConfigReloadRef.current = true;
      onConfigChange(finalConfig);
      setIsConfigured(true);
      saveDatasetToHistory(finalConfig, 'preprocessed', previewData?.columns || []);

    } catch (error) {
      onError('Failed to remove duplicates: ' + error.message);
    } finally {
      dispatchDuplicates({ type: 'SET_REMOVING', payload: false });
    }
  };

  /**
   * Handle user choosing to keep duplicates and continue
   */
  const handleKeepDuplicates = async () => {
    dispatchDuplicates({ type: 'HIDE_MODAL' });
    
    // Continue with the rest of the loading process
    const previewData = preview;
    
    // Check for missing columns
    const missing = checkMissingColumns(previewData);
    
    if (missing.length > 0) {
      dispatchMissingCols({ type: 'SET_MISSING_COLUMNS', payload: missing });
      dispatchMissingCols({ type: 'SHOW_MODAL' });
      return;
    }

    // All columns present - configure
    const uploadedPath = uploadedFilePath || dataFiles;
    const fileFormat = getFileFormat(uploadedFileName || uploadedPath);
    
    const finalConfig = {
      data_files: uploadedPath,
      file_format: fileFormat,
      num_samples: numSamples || null,
      shuffle,
      seed,
      uploaded_file: uploadedFileName
    };

    skipNextConfigReloadRef.current = true;
    onConfigChange(finalConfig);
    setIsConfigured(true);
    saveDatasetToHistory(finalConfig, 'preprocessed', previewData?.columns || []);
  };

  /**
   * Handle dataset loading from manual path
   */
  const handleLoadDataset = async () => {
    try {
      setIsLoading(true);

      // Detect file format from path
      const fileFormat = getFileFormat(dataFiles);

      // Load the dataset WITH the user's specified filters
      const loadConfig = {
        data_files: dataFiles,
        file_format: fileFormat,  // Pass format for optimal pandas loading
        num_samples: numSamples || null, // Use user's filter (or null for all samples)
        shuffle,
        seed,
      };

      // Try to load dataset in backend (works for existing flows, may fail for custom)
      let previewColumns = [];
      try {
        const response = await datasetAPI.loadDataset(loadConfig);
        
        // Get preview of the FILTERED dataset
        const previewData = await datasetAPI.getPreview();
        setPreview(previewData);
        previewColumns = previewData?.columns || [];
      } catch (apiError) {
        // For custom flows, backend load may fail - that's okay
        // The dataset will be loaded when the flow is actually run
        console.warn('Dataset load API failed (expected for custom flows):', apiError);
      }

      // Save the configuration to parent state
      const finalConfig = {
        data_files: dataFiles,
        file_format: fileFormat,
        num_samples: numSamples || null, // Use user's filter
        shuffle,
        seed,
      };

      // Skip the next config reload since we just configured the dataset
      skipNextConfigReloadRef.current = true;
      
      // Update parent state
      onConfigChange(finalConfig);
      setIsConfigured(true);
      
      // Save to history (pass columns directly since state may not be updated yet)
      saveDatasetToHistory({ ...finalConfig, uploaded_file: uploadedFileName }, 'preprocessed', previewColumns);

    } catch (error) {
      onError('Failed to load dataset: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Check if form is valid
   */
  const isFormValid = () => {
    // Valid if we have file content OR if we have a filename from existing config
    return (uploadedFile || uploadedFileName || dataFiles) && split;
  };

  // NOTE: loading and !selectedFlow early returns moved after all hooks (below)
  // to maintain consistent hook order across renders

  /**
   * Handle user choosing to add missing columns with repetitive format
   */
  const handleUseRepetitiveFormat = () => {
    dispatchMissingCols({ type: 'START_ADDING' });
  };

  /**
   * Handle user canceling and choosing to fix file manually
   */
  const handleCancelAndFixManually = () => {
    dispatchMissingCols({ type: 'CANCEL_FIX_MANUALLY' });
    // Clear the uploaded file so they can upload a fixed version
    handleClearUpload();
  };

  /**
   * Save value for a missing column
   */
  const handleSaveMissingColumnValue = (columnName, value) => {
    const newValues = {
      ...missingColumnValues,
      [columnName]: value
    };
    dispatchMissingCols({ type: 'SAVE_COLUMN_VALUE', columnName, value });
    
    // Move to next column or finish
    if (currentMissingColumnIndex < missingColumns.length - 1) {
      dispatchMissingCols({ type: 'NEXT_COLUMN' });
    } else {
      // All columns filled - apply to dataset and continue loading
      applyMissingColumnsAndLoad(newValues);
    }
  };

  /**
   * Apply missing column values to dataset and continue loading
   */
  const applyMissingColumnsAndLoad = async (columnValues) => {
    try {
      dispatchMissingCols({ type: 'SET_ADDING', payload: false });
      setIsLoading(true);
      
      // Detect file format from filename
      const uploadedPath = uploadedFilePath || dataFiles;
      const fileFormat = getFileFormat(uploadedFileName || uploadedPath);
      
      // Reload the dataset WITH the added columns
      const loadConfig = {
        data_files: uploadedPath,
        file_format: fileFormat,
        num_samples: numSamples || null,
        shuffle,
        seed,
        added_columns: columnValues, // Pass columns to add to backend
      };

      // Call the backend to reload dataset with added columns
      await datasetAPI.loadDataset(loadConfig);
      
      // Get updated preview (should now include the new columns)
      const previewData = await datasetAPI.getPreview();
      setPreview(previewData);
      
      const finalConfig = {
        data_files: uploadedPath,
        file_format: fileFormat,
        num_samples: numSamples || null,
        shuffle,
        seed,
        uploaded_file: uploadedFileName,
        added_columns: columnValues, // Track which columns were added
      };

      // Skip the next config reload since we just configured the dataset
      skipNextConfigReloadRef.current = true;
      
      // Update parent state
      if (onConfigChange) {
        onConfigChange(finalConfig);
      }
      setIsConfigured(true);
      
      // Save to history (with updated columns)
      saveDatasetToHistory(finalConfig, 'preprocessed', previewData?.columns || []);
      
      // Reset missing columns state
      dispatchMissingCols({ type: 'RESET' });

    } catch (error) {
      onError('Failed to apply missing columns: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Get required columns for the selected flow
  const requiredColumns = schema?.required_columns || schema?.requirements?.required_columns || [];
  
  // Handle dataset created from PDF preprocessing
  const handlePDFDatasetCreated = async (datasetInfo) => {
    // Set the file path and trigger reload
    setDataFiles(datasetInfo.file_path);
    dispatchUpload({ type: 'SET_FILE_PATH', payload: datasetInfo.file_path });
    dispatchUpload({ type: 'SET_FILE_NAME', payload: datasetInfo.file_path.split('/').pop() });
    setNumSamples(datasetInfo.num_records);
    
    // Switch back to preprocessed view to show the loaded dataset
    // But when skipSourceSelection is true, don't change the source type
    // (the wizard manages source via separate steps and changing it would remove the PDF step)
    if (!skipSourceSelection) {
      setDatasetSource('preprocessed');
    }
    
    // Auto-load the dataset
    try {
      const loadConfig = {
        data_files: datasetInfo.file_path,
        file_format: 'jsonl',
        num_samples: datasetInfo.num_records,
        shuffle: shuffle,
        seed: seed,
      };
      
      await datasetAPI.loadDataset(loadConfig);
      const previewData = await datasetAPI.getPreview();
      setPreview(previewData);
      
      const finalConfig = {
        ...loadConfig,
        uploaded_file: datasetInfo.file_path.split('/').pop(),
      };
      
      // Skip the next config reload since we just configured the dataset
      skipNextConfigReloadRef.current = true;
      
      onConfigChange(finalConfig);
      setIsConfigured(true);
      
      // Save to history (pass columns directly since state may not be updated yet)
      saveDatasetToHistory(finalConfig, 'pdf', previewData?.columns || []);
    } catch (error) {
      console.error('Error loading preprocessed dataset:', error);
    }
  };

  // Auto-load PDF dataset when navigating to this step after PDF preprocessing
  // This handles the case where PDF preprocessing is a separate wizard step
  useEffect(() => {
    if (skipSourceSelection && externalPdfDatasetInfo?.file_path && datasetSource === 'pdf' && !isConfigured) {
      handlePDFDatasetCreated(externalPdfDatasetInfo);
    }
  }, [externalPdfDatasetInfo, skipSourceSelection]);

  // Early returns (placed after ALL hooks to maintain consistent hook order)
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem' }}>
        <Spinner size="xl" />
        <div style={{ marginTop: '1rem' }}>Loading dataset requirements...</div>
      </div>
    );
  }

  if (!selectedFlow) {
    return (
      <Alert
        variant={AlertVariant.warning}
        isInline
        title="No flow selected"
      >
        Please select a flow in the first step before configuring the dataset.
      </Alert>
    );
  }

  // Force source selection mode: only show the source cards (used as a separate wizard step)
  if (forceSourceSelection) {
    return (
      <div style={{ padding: '1rem' }}>
        <Title headingLevel="h2" size="xl" style={{ marginBottom: '0.5rem' }}>
          Choose Dataset Source
        </Title>
        <p style={{ marginBottom: '1.5rem', color: '#6a6e73' }}>
          Select how you want to provide data for this flow.
        </p>
        
        {/* Required Columns Section */}
        <Card style={{ marginBottom: '1.5rem', backgroundColor: '#f0f0f0' }}>
          <CardBody style={{ padding: '1rem 1.5rem' }}>
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.75rem' }}>
              Required Columns for this Flow
            </Title>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {requiredColumns.map(col => (
                <span 
                  key={col}
                  style={{ 
                    backgroundColor: '#fff',
                    border: '1px solid #d2d2d2',
                    borderRadius: '4px',
                    padding: '0.25rem 0.75rem',
                    fontSize: '0.875rem',
                    fontFamily: 'monospace',
                    color: '#151515'
                  }}
                >
                  {col}
                </span>
              ))}
            </div>
          </CardBody>
        </Card>
        
        <Grid hasGutter>
          {/* Reuse Previous Dataset */}
          {previousDatasets.length > 0 && (
            <GridItem span={4}>
              <Card 
                isSelectable 
                isSelected={datasetSource === 'reuse'}
                onClick={() => setDatasetSource('reuse')}
                style={{ 
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  border: datasetSource === 'reuse' ? '2px solid #3e8635' : '2px solid transparent',
                }}
                onMouseEnter={(e) => { if (datasetSource !== 'reuse') e.currentTarget.style.borderColor = '#3e8635'; }}
                onMouseLeave={(e) => { if (datasetSource !== 'reuse') e.currentTarget.style.borderColor = 'transparent'; }}
              >
                <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                    <div style={{ 
                      width: '64px', height: '64px', borderRadius: '50%', backgroundColor: '#e8f5e9', 
                      display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto'
                    }}>
                      <HistoryIcon style={{ fontSize: '2rem', color: '#3e8635' }} />
                    </div>
                    <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                      Reuse a previous dataset
                    </Title>
                  </div>
                  <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '1rem', flex: 1, marginBottom: '1rem' }}>
                    <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Available datasets:</p>
                    <List isPlain style={{ fontSize: '0.875rem' }}>
                      {previousDatasets.slice(0, 3).map((ds, idx) => (
                        <ListItem key={ds.id || idx}>
                          ✓ {ds.name} <span style={{ color: '#6a6e73', fontSize: '0.75rem' }}>({ds.source === 'pdf' ? 'PDF' : 'Uploaded'})</span>
                        </ListItem>
                      ))}
                      {previousDatasets.length > 3 && (
                        <ListItem style={{ color: '#6a6e73' }}>+{previousDatasets.length - 3} more...</ListItem>
                      )}
                    </List>
                  </div>
                  <Button variant={datasetSource === 'reuse' ? 'primary' : 'secondary'} isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('reuse'); }}>
                    {datasetSource === 'reuse' ? '✓ Selected' : 'Choose from History'}
                  </Button>
                </CardBody>
              </Card>
            </GridItem>
          )}
          
          {/* Upload preprocessed dataset */}
          <GridItem span={previousDatasets.length > 0 ? 4 : 6}>
            <Card 
              isSelectable 
              isSelected={datasetSource === 'preprocessed'}
              onClick={() => setDatasetSource('preprocessed')}
              style={{ 
                height: '100%',
                cursor: 'pointer',
                transition: 'all 0.2s',
                border: datasetSource === 'preprocessed' ? '2px solid #0066cc' : '2px solid transparent',
              }}
              onMouseEnter={(e) => { if (datasetSource !== 'preprocessed') e.currentTarget.style.borderColor = '#0066cc'; }}
              onMouseLeave={(e) => { if (datasetSource !== 'preprocessed') e.currentTarget.style.borderColor = 'transparent'; }}
            >
              <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                  <div style={{ 
                    width: '64px', height: '64px', borderRadius: '50%', backgroundColor: '#e7f1fa', 
                    display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto'
                  }}>
                    <FileIcon style={{ fontSize: '2rem', color: '#0066cc' }} />
                  </div>
                  <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                    Upload a preprocessed dataset from local computer
                  </Title>
                </div>
                <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '1rem', flex: 1, marginBottom: '1rem' }}>
                  <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>What you need:</p>
                  <List isPlain style={{ fontSize: '0.875rem' }}>
                    <ListItem>✓ A JSONL, JSON, CSV, or Parquet file</ListItem>
                    <ListItem>✓ If any required columns are missing, you will be asked to fill them in the UI</ListItem>
                  </List>
                </div>
                <Button variant={datasetSource === 'preprocessed' ? 'primary' : 'secondary'} isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('preprocessed'); }}>
                  {datasetSource === 'preprocessed' ? '✓ Selected' : 'Upload Dataset File'}
                </Button>
              </CardBody>
            </Card>
          </GridItem>
          
          {/* Upload files and preprocess */}
          <GridItem span={previousDatasets.length > 0 ? 4 : 6}>
            <Card 
              isSelectable 
              isSelected={datasetSource === 'pdf'}
              onClick={() => setDatasetSource('pdf')}
              style={{ 
                height: '100%',
                cursor: 'pointer',
                transition: 'all 0.2s',
                border: datasetSource === 'pdf' ? '2px solid #c9190b' : '2px solid transparent',
              }}
              onMouseEnter={(e) => { if (datasetSource !== 'pdf') e.currentTarget.style.borderColor = '#c9190b'; }}
              onMouseLeave={(e) => { if (datasetSource !== 'pdf') e.currentTarget.style.borderColor = 'transparent'; }}
            >
              <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                  <div style={{ 
                    width: '64px', height: '64px', borderRadius: '50%', backgroundColor: '#fce8e8', 
                    display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto'
                  }}>
                    <FilePdfIcon style={{ fontSize: '2rem', color: '#c9190b' }} />
                  </div>
                  <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                    Upload files and preprocess
                  </Title>
                </div>
                <div style={{ backgroundColor: '#f5f5f5', borderRadius: '6px', padding: '1rem', flex: 1, marginBottom: '1rem' }}>
                  <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Preprocessing pipeline:</p>
                  <List isPlain style={{ fontSize: '0.875rem' }}>
                    <ListItem>1. Upload PDF or Markdown files</ListItem>
                    <ListItem>2. Convert PDFs to Markdown (using docling)</ListItem>
                    <ListItem>3. Chunk documents</ListItem>
                    <ListItem>4. Add required columns (ICL, domain, etc.)</ListItem>
                  </List>
                </div>
                <Button variant={datasetSource === 'pdf' ? 'primary' : 'secondary'} isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('pdf'); }}>
                  {datasetSource === 'pdf' ? '✓ Selected' : 'Start Preprocessing'}
                </Button>
              </CardBody>
            </Card>
          </GridItem>
        </Grid>
        
        {datasetSource !== 'none' && (
          <Alert variant="info" isInline title="Press Next to continue" style={{ marginTop: '1.5rem' }}>
            You selected <strong>{datasetSource === 'pdf' ? 'Upload files and preprocess' : datasetSource === 'preprocessed' ? 'Upload preprocessed dataset' : 'Reuse a previous dataset'}</strong>. Press <strong>Next</strong> to continue.
          </Alert>
        )}
      </div>
    );
  }

  // Source selection UI (original - used when component is rendered standalone without forceSourceSelection)
  // Skip this when skipSourceSelection is true (source was chosen in a separate wizard step)
  if (!skipSourceSelection && datasetSource === 'none' && !importedConfig && !datasetConfig?.data_files) {
    return (
      <div style={{ padding: '1rem' }}>
        <Title headingLevel="h2" size="xl" style={{ marginBottom: '0.5rem' }}>
          Choose Dataset Source
        </Title>
        <p style={{ marginBottom: '1.5rem', color: '#6a6e73' }}>
          Select how you want to provide data for this flow.
        </p>
        
        {/* Required Columns Section - Shared between both options */}
        <Card style={{ marginBottom: '1.5rem', backgroundColor: '#f0f0f0' }}>
          <CardBody style={{ padding: '1rem 1.5rem' }}>
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.75rem' }}>
              Required Columns for this Flow
            </Title>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {requiredColumns.map(col => (
                <span 
                  key={col}
                  style={{ 
                    backgroundColor: '#fff',
                    border: '1px solid #d2d2d2',
                    borderRadius: '4px',
                    padding: '0.25rem 0.75rem',
                    fontSize: '0.875rem',
                    fontFamily: 'monospace',
                    color: '#151515'
                  }}
                >
                  {col}
                </span>
              ))}
            </div>
          </CardBody>
        </Card>
        
        <Grid hasGutter>
          {/* Option 0: Reuse Previous Dataset (only show if there are previous datasets) */}
          {previousDatasets.length > 0 && (
            <GridItem span={4}>
              <Card 
                isSelectable 
                isSelected={false}
                onClick={() => setDatasetSource('reuse')}
                style={{ 
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  border: '2px solid transparent',
                }}
                onMouseEnter={(e) => e.currentTarget.style.borderColor = '#3e8635'}
                onMouseLeave={(e) => e.currentTarget.style.borderColor = 'transparent'}
              >
                <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                    <div style={{ 
                      width: '64px', 
                      height: '64px', 
                      borderRadius: '50%', 
                      backgroundColor: '#e8f5e9', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      margin: '0 auto 1rem auto'
                    }}>
                      <HistoryIcon style={{ fontSize: '2rem', color: '#3e8635' }} />
                    </div>
                    <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                      Reuse a previous dataset
                    </Title>
                  </div>
                  
                  <div style={{ 
                    backgroundColor: '#f5f5f5', 
                    borderRadius: '6px', 
                    padding: '1rem', 
                    flex: 1,
                    marginBottom: '1rem'
                  }}>
                    <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Available datasets:</p>
                    <List isPlain style={{ fontSize: '0.875rem' }}>
                      {previousDatasets.slice(0, 3).map((ds, idx) => (
                        <ListItem key={ds.id || idx}>
                          ✓ {ds.name} <span style={{ color: '#6a6e73', fontSize: '0.75rem' }}>({ds.source === 'pdf' ? 'PDF' : 'Uploaded'})</span>
                        </ListItem>
                      ))}
                      {previousDatasets.length > 3 && (
                        <ListItem style={{ color: '#6a6e73' }}>
                          +{previousDatasets.length - 3} more...
                        </ListItem>
                      )}
                    </List>
                  </div>
                  
                  <Button variant="primary" isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('reuse'); }}>
                    Choose from History
                  </Button>
                </CardBody>
              </Card>
            </GridItem>
          )}
          
          {/* Option 1: Upload preprocessed from local computer */}
          <GridItem span={previousDatasets.length > 0 ? 4 : 6}>
            <Card 
              isSelectable 
              isSelected={false}
              onClick={() => setDatasetSource('preprocessed')}
              style={{ 
                height: '100%',
                cursor: 'pointer',
                transition: 'all 0.2s',
                border: '2px solid transparent',
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = '#0066cc'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = 'transparent'}
            >
              <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                  <div style={{ 
                    width: '64px', 
                    height: '64px', 
                    borderRadius: '50%', 
                    backgroundColor: '#e7f1fa', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    margin: '0 auto 1rem auto'
                  }}>
                    <FileIcon style={{ fontSize: '2rem', color: '#0066cc' }} />
                  </div>
                  <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                    Upload a preprocessed dataset from local computer
                  </Title>
                </div>
                
                <div style={{ 
                  backgroundColor: '#f5f5f5', 
                  borderRadius: '6px', 
                  padding: '1rem', 
                  flex: 1,
                  marginBottom: '1rem'
                }}>
                  <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>What you need:</p>
                  <List isPlain style={{ fontSize: '0.875rem' }}>
                    <ListItem>✓ A JSONL, JSON, CSV, or Parquet file</ListItem>
                    <ListItem>✓ If any required columns are missing, you will be asked to fill them in the UI</ListItem>
                  </List>
                </div>
                
                <Button variant="primary" isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('preprocessed'); }}>
                  Upload Dataset File
                </Button>
              </CardBody>
            </Card>
          </GridItem>
          
          {/* Option 2: Upload files and preprocess */}
          <GridItem span={previousDatasets.length > 0 ? 4 : 6}>
            <Card 
              isSelectable 
              isSelected={false}
              onClick={() => setDatasetSource('pdf')}
              style={{ 
                height: '100%',
                cursor: 'pointer',
                transition: 'all 0.2s',
                border: '2px solid transparent',
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = '#c9190b'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = 'transparent'}
            >
              <CardBody style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                  <div style={{ 
                    width: '64px', 
                    height: '64px', 
                    borderRadius: '50%', 
                    backgroundColor: '#fce8e8', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    margin: '0 auto 1rem auto'
                  }}>
                    <FilePdfIcon style={{ fontSize: '2rem', color: '#c9190b' }} />
                  </div>
                  <Title headingLevel="h3" size="lg" style={{ marginBottom: '0.5rem' }}>
                    Upload files and preprocess
                  </Title>
                </div>
                
                <div style={{ 
                  backgroundColor: '#f5f5f5', 
                  borderRadius: '6px', 
                  padding: '1rem', 
                  flex: 1,
                  marginBottom: '1rem'
                }}>
                  <p style={{ fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Preprocessing pipeline:</p>
                  <List isPlain style={{ fontSize: '0.875rem' }}>
                    <ListItem>1. Upload PDF or Markdown files</ListItem>
                    <ListItem>2. Convert PDFs to Markdown (using docling)</ListItem>
                    <ListItem>3. Chunk documents</ListItem>
                    <ListItem>4. Add required columns (ICL, domain, etc.)</ListItem>
                  </List>
                </div>
                
                <Button variant="primary" isBlock onClick={(e) => { e.stopPropagation(); setDatasetSource('pdf'); }}>
                  Start Preprocessing
                </Button>
              </CardBody>
            </Card>
          </GridItem>
        </Grid>
      </div>
    );
  }
  
  // PDF Preprocessing Mode
  // When skipSourceSelection is set and source is 'pdf', the preprocessing is handled by a separate
  // wizard step. Fall through to the dataset configuration view below (auto-load from pdfDatasetInfo).
  if (datasetSource === 'pdf' && !skipSourceSelection) {
    return (
      <PDFPreprocessingStep
        requiredColumns={requiredColumns}
        onDatasetCreated={(info) => {
          // Notify parent wizard about the created dataset (for state persistence)
          if (onPdfDatasetCreated) {
            onPdfDatasetCreated(info);
          }
          handlePDFDatasetCreated(info);
        }}
        onCancel={() => setDatasetSource('none')}
        onError={onError}
        savedState={pdfPreprocessingState}
        onStateChange={(state) => setPdfPreprocessingState(state)}
      />
    );
  }
  
  // Reuse Previous Dataset Mode
  if (datasetSource === 'reuse') {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
          {!skipSourceSelection && (
            <Button 
              variant="link" 
              onClick={() => setDatasetSource('none')}
              style={{ fontSize: '1rem', fontWeight: '500', paddingLeft: 0 }}
            >
              ← Change Dataset Source
            </Button>
          )}
        </div>
        
        <Title headingLevel="h2" size="xl" style={{ marginBottom: '0.5rem' }}>
          Select a Previous Dataset
        </Title>
        <p style={{ marginBottom: '1.5rem', color: '#6a6e73' }}>
          Choose from datasets you've previously configured or preprocessed.
        </p>
        
        {/* Required Columns Section */}
        <Card style={{ marginBottom: '1.5rem', backgroundColor: '#f0f0f0' }}>
          <CardBody style={{ padding: '1rem 1.5rem' }}>
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.75rem' }}>
              Required Columns for this Flow
            </Title>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {requiredColumns.map(col => (
                <span 
                  key={col}
                  style={{ 
                    backgroundColor: '#fff',
                    border: '1px solid #d2d2d2',
                    borderRadius: '4px',
                    padding: '0.25rem 0.75rem',
                    fontSize: '0.875rem',
                    fontFamily: 'monospace',
                    color: '#151515'
                  }}
                >
                  {col}
                </span>
              ))}
            </div>
          </CardBody>
        </Card>
        
        {/* Dataset List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {previousDatasets.map((dataset) => {
            const isSelected = selectedPreviousDataset?.id === dataset.id;
            const dateStr = dataset.timestamp ? new Date(dataset.timestamp).toLocaleDateString() : 'Unknown date';
            
            return (
              <Card
                key={dataset.id}
                isSelectable
                isSelected={isSelected}
                onClick={() => {
                  // Select the card and switch to config view (num_samples, shuffle, Load Dataset)
                  setSelectedPreviousDataset(dataset);

                  // Set file info from the selected dataset
                  setDataFiles(dataset.data_files);
                  dispatchUpload({ type: 'SET_FILE_PATH', payload: dataset.data_files });
                  dispatchUpload({ type: 'SET_FILE_NAME', payload: dataset.name });
                  if (dataset.num_samples) {
                    setNumSamples(dataset.num_samples);
                  }

                  // Mark as from reuse — config view will show "Load Dataset" not "Reload"
                  setIsFromReuse(true);
                  setIsConfigured(false);

                  // Switch to preprocessed config view (same step, not next step)
                  setDatasetSource('preprocessed');
                }}
                style={{
                  cursor: 'pointer',
                  border: isSelected ? '2px solid #3e8635' : '1px solid #d2d2d2',
                  backgroundColor: isSelected ? '#f0fff0' : '#fff',
                }}
              >
                <CardBody style={{ padding: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                        {dataset.source === 'pdf' ? (
                          <FilePdfIcon style={{ color: '#c9190b' }} />
                        ) : (
                          <FileIcon style={{ color: '#0066cc' }} />
                        )}
                        <span style={{ fontWeight: 'bold', fontSize: '1rem' }}>{dataset.name}</span>
                        <span style={{ 
                          backgroundColor: dataset.source === 'pdf' ? '#fce8e8' : '#e7f1fa',
                          color: dataset.source === 'pdf' ? '#c9190b' : '#0066cc',
                          padding: '0.125rem 0.5rem',
                          borderRadius: '10px',
                          fontSize: '0.75rem',
                          fontWeight: '500'
                        }}>
                          {dataset.source === 'pdf' ? 'From PDF' : 'Uploaded'}
                        </span>
                      </div>
                      <div style={{ color: '#6a6e73', fontSize: '0.875rem' }}>
                        <span>Used with: {dataset.flow}</span>
                        <span style={{ margin: '0 0.5rem' }}>•</span>
                        <span>{dateStr}</span>
                        {dataset.num_samples && (
                          <>
                            <span style={{ margin: '0 0.5rem' }}>•</span>
                            <span>{dataset.num_samples} samples</span>
                          </>
                        )}
                      </div>
                      {dataset.columns && dataset.columns.length > 0 && (
                        <div style={{ marginTop: '0.5rem', display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                          {dataset.columns.slice(0, 5).map(col => (
                            <span 
                              key={col}
                              style={{ 
                                backgroundColor: '#f5f5f5',
                                border: '1px solid #e0e0e0',
                                borderRadius: '3px',
                                padding: '0.125rem 0.375rem',
                                fontSize: '0.75rem',
                                fontFamily: 'monospace',
                              }}
                            >
                              {col}
                            </span>
                          ))}
                          {dataset.columns.length > 5 && (
                            <span style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                              +{dataset.columns.length - 5} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    {isSelected && (
                      <CheckCircleIcon style={{ color: '#3e8635', fontSize: '1.5rem' }} />
                    )}
                  </div>
                </CardBody>
              </Card>
            );
          })}
        </div>
        
        {/* Back button */}
        <div style={{ marginTop: '1.5rem' }}>
          {!skipSourceSelection && (
            <Button 
              variant="secondary"
              onClick={() => {
                setSelectedPreviousDataset(null);
                setDatasetSource('none');
              }}
            >
              Cancel
            </Button>
          )}
        </div>
      </div>
    );
  }
  
  // Auto-set to preprocessed mode if we have existing config
  if (datasetSource === 'none' && (importedConfig || datasetConfig?.data_files)) {
    setDatasetSource('preprocessed');
  }

  // When skipSourceSelection is true and datasetSource is 'pdf' but dataset hasn't loaded yet, 
  // show a loading state while the auto-load useEffect runs
  if (skipSourceSelection && datasetSource === 'pdf' && !isConfigured && !isLoading && !dataFiles) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <Spinner size="lg" />
        <p style={{ marginTop: '1rem', color: '#6a6e73' }}>Loading preprocessed dataset...</p>
      </div>
    );
  }

  return (
    <>
    <Grid hasGutter style={{ height: '100%' }}>
      {/* Import Success Indicator */}
      {importedConfig && (
        <GridItem span={12}>
          <Alert
            variant={AlertVariant.success}
            isInline
            title="Dataset configuration loaded from import"
          >
            <p>
              ✅ Dataset settings have been pre-filled: <strong>{importedConfig.data_files}</strong>
            </p>
          </Alert>
        </GridItem>
      )}
      
      {/* Option to change dataset source - shown at top left */}
      {!importedConfig && (
        <GridItem span={12}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'flex-start',
            gap: '1rem'
          }}>
            {!skipSourceSelection && (
              <Button 
                variant="link" 
                onClick={() => {
                  // Reset states and go back to source selection
                  setDatasetSource('none');
                  setIsConfigured(false);
                  dispatchUpload({ type: 'SET_FILE_NAME', payload: '' });
                  dispatchUpload({ type: 'SET_FILE_PATH', payload: '' });
                  setDataFiles('');
                  setPreview(null);
                  // Clear parent config so source selection shows
                  onConfigChange({});
                }}
                style={{ fontSize: '1rem', fontWeight: '500', paddingLeft: 0 }}
              >
                ← Change Dataset Source
              </Button>
            )}
            {!skipSourceSelection && pdfPreprocessingState && pdfPreprocessingState.jobId && (
              <Button 
                variant="link" 
                onClick={() => setDatasetSource('pdf')}
                style={{ fontSize: '1rem', fontWeight: '500' }}
              >
                Return to PDF Preprocessing
              </Button>
            )}
          </div>
        </GridItem>
      )}

      {/* Left Panel - Configuration Form */}
      <GridItem span={7} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Dataset Configuration
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto', padding: '1rem' }}>
            <Form>
              {/* Upload File - Always Upload Mode */}
                <FormGroup 
                  label="Upload Dataset File" 
                  isRequired 
                  fieldId="file-upload"
                helperText="Supports JSONL, JSON, CSV, and Parquet formats"
                style={{ marginBottom: '1rem' }}
                >
                <div style={{
                  border: '2px dashed #d2d2d2',
                  borderRadius: '4px',
                  padding: '2rem',
                  textAlign: 'center',
                  backgroundColor: uploadedFileName ? '#f0f9ff' : '#fafafa',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = '#0066cc';
                  e.currentTarget.style.backgroundColor = '#f0f9ff';
                }}
                onDragLeave={(e) => {
                  e.currentTarget.style.borderColor = '#d2d2d2';
                  e.currentTarget.style.backgroundColor = uploadedFileName ? '#f0f9ff' : '#fafafa';
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = '#d2d2d2';
                  const file = e.dataTransfer.files[0];
                  if (file) {
                    handleFileUpload(e, file);
                  }
                }}
                >
                  {uploadedFileName ? (
                    <div>
                      <CheckCircleIcon style={{ fontSize: '3rem', color: '#3e8635', marginBottom: '1rem' }} />
                      <div style={{ fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                        {uploadedFile ? 'File uploaded: ' : isFromReuse ? 'Dataset file: ' : 'Referenced file: '}{uploadedFileName}
                      </div>
                      {!uploadedFile && uploadedFilePath && !isFromReuse && (
                        <div style={{ fontSize: '0.9rem', color: '#3e8635', marginBottom: '0.5rem' }}>
                          ✓ Dataset loaded from configuration
                        </div>
                      )}
                      {isFromReuse && !isConfigured && (
                        <div style={{ fontSize: '0.9rem', color: '#0066cc', marginBottom: '0.5rem' }}>
                          Dataset selected — click Load Dataset below to load it
                        </div>
                      )}
                      <div style={{ fontSize: '0.9rem', color: '#6a6e73', marginBottom: '1rem' }}>
                        Ready to load with the parameters below
                      </div>
                      <Button variant="secondary" size="sm" onClick={handleClearUpload}>
                        Remove File
                      </Button>
                    </div>
                  ) : (
                    <div>
                      <UploadIcon style={{ fontSize: '3rem', color: '#6a6e73', marginBottom: '1rem' }} />
                      <div style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>
                        Drag and drop a dataset file here
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                        Supports: JSONL, JSON, CSV, Parquet
                      </div>
                      <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginBottom: '1rem' }}>
                        or
                      </div>
                      <input
                        type="file"
                        accept=".jsonl,.json,.csv,.parquet,.pq"
                        style={{ display: 'none' }}
                        id="file-input-hidden"
                        onChange={(e) => {
                          if (e.target.files[0]) {
                            handleFileUpload(e, e.target.files[0]);
                          }
                        }}
                      />
                      <Button
                        variant="primary"
                        onClick={() => document.getElementById('file-input-hidden').click()}
                      >
                        Browse
                      </Button>
                    </div>
                  )}
                </div>
                </FormGroup>

              {/* Compact Grid Layout for smaller fields */}
              <Grid hasGutter style={{ marginBottom: '0.75rem' }}>
                {/* Number of Samples */}
                <GridItem span={6}>
                  <FormGroup 
                    label={
                      <Tooltip content="The number of data samples to use from your dataset. Use a smaller number for testing.">
                        <span style={{ cursor: 'help', borderBottom: '1px dashed #6a6e73', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                          Number of Samples <OutlinedQuestionCircleIcon style={{ fontSize: '0.875rem', color: '#6a6e73' }} />
                        </span>
                      </Tooltip>
                    }
                    fieldId="num-samples"
                    helperText="Adjust as needed before loading"
                  >
                    <NumberInput
                      id="num-samples"
                      value={numSamples}
                      onMinus={() => setNumSamples(Math.max(1, numSamples - 1))}
                      onPlus={() => setNumSamples(numSamples + 1)}
                      onChange={(event) => {
                        const value = parseInt(event.target.value, 10);
                        setNumSamples(isNaN(value) ? 0 : value);
                      }}
                      min={1}
                      widthChars={8}
                    />
                  </FormGroup>
                </GridItem>

                {/* Shuffle */}
                <GridItem span={6}>
                  <FormGroup fieldId="shuffle">
                    <Tooltip content="Randomize the order of samples. Recommended for training data to prevent ordering bias.">
                      <span style={{ cursor: 'help' }}>
                        <Checkbox
                          id="shuffle"
                          label={
                            <span style={{ borderBottom: '1px dashed #6a6e73', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                              Shuffle dataset <OutlinedQuestionCircleIcon style={{ fontSize: '0.875rem', color: '#6a6e73' }} />
                            </span>
                          }
                          isChecked={shuffle}
                          onChange={(event, checked) => setShuffle(checked)}
                        />
                      </span>
                    </Tooltip>
                  </FormGroup>
                </GridItem>

                {/* Seed (only shown if shuffle is enabled) */}
                {shuffle && (
                  <GridItem span={6}>
                    <FormGroup 
                      label={
                        <Tooltip content="A fixed number to ensure reproducible shuffling. Same seed = same order each time.">
                          <span style={{ cursor: 'help', borderBottom: '1px dashed #6a6e73', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                            Random Seed <OutlinedQuestionCircleIcon style={{ fontSize: '0.875rem', color: '#6a6e73' }} />
                          </span>
                        </Tooltip>
                      }
                      fieldId="seed"
                    >
                      <NumberInput
                        id="seed"
                        value={seed}
                        onMinus={() => setSeed(Math.max(0, seed - 1))}
                        onPlus={() => setSeed(seed + 1)}
                        onChange={(event) => {
                          const value = parseInt(event.target.value, 10);
                          setSeed(isNaN(value) ? 42 : value);
                        }}
                        min={0}
                        widthChars={8}
                      />
                    </FormGroup>
                  </GridItem>
                )}
              </Grid>

              {/* Load Button - at bottom */}
              <div style={{ 
                marginTop: '1rem',
                paddingTop: '1rem',
                borderTop: '1px solid #d2d2d2',
                display: 'flex',
                gap: '1rem',
                alignItems: 'center'
              }}>
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleLoadUploadedDataset}
                  isDisabled={!isFormValid()}
                  isLoading={isLoading}
                >
                  {isConfigured ? 'Reload Dataset' : 'Load Dataset'}
                </Button>
                
                {!isConfigured && uploadedFilePath && (
                  <Alert
                    variant={AlertVariant.info}
                    isInline
                    title={isFromReuse 
                      ? "Dataset selected — adjust parameters if needed, then click Load Dataset" 
                      : "File uploaded - adjust parameters above, then click Load Dataset"}
                    style={{ margin: 0 }}
                  />
                )}
                
                {isConfigured && (
                  <Alert
                    variant={AlertVariant.success}
                    isInline
                    title="Dataset loaded and configured"
                    style={{ margin: 0 }}
                  />
                )}
              </div>

              {/* Dataset Preview - Button Only */}
              {preview && preview.preview && (
                <div style={{ marginTop: '1rem' }}>
                  <Button
                    variant="secondary"
                    onClick={() => setIsPreviewExpanded(!isPreviewExpanded)}
                    icon={isPreviewExpanded ? undefined : <CheckCircleIcon />}
                  >
                    {isPreviewExpanded ? 'Hide Preview' : 'See Preview'} ({preview.preview_size || 0} of {preview.num_samples} samples)
                  </Button>
                  
                  {isPreviewExpanded && (
                    <div style={{ 
                      marginTop: '1rem',
                      maxHeight: '500px',
                      overflowY: 'auto',
                      border: '1px solid #d2d2d2',
                      borderRadius: '4px',
                      padding: '1rem',
                      backgroundColor: '#f5f5f5'
                    }}>
                      {(() => {
                        // Get column names from preview.columns (the actual column names)
                        const columnNames = preview.columns || [];
                        
                        // Backend returns column-oriented data: { col1: [val1, val2, ...], col2: [val1, val2, ...] }
                        // We need to transform to row-oriented: [ {col1: val1, col2: val1}, {col1: val2, col2: val2}, ... ]
                        const previewObj = preview.preview || {};
                        const numSamples = preview.preview_size || 0;
                        
                        // Create array of sample objects
                        const samples = [];
                        for (let i = 0; i < numSamples; i++) {
                          const sample = {};
                          columnNames.forEach(col => {
                            if (previewObj[col] && previewObj[col][i] !== undefined) {
                              sample[col] = previewObj[col][i];
                            }
                          });
                          samples.push(sample);
                        }
                        
                        return samples.map((sample, idx) => {
                          const selectedColumn = selectedColumnsPerSample[idx] || (columnNames.length > 0 ? columnNames[0] : null);
                          
                          return (
                            <div key={idx} style={{ 
                              marginBottom: idx < samples.length - 1 ? '1.5rem' : 0,
                              padding: '1rem',
                              backgroundColor: 'white',
                              borderRadius: '8px',
                              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                              fontSize: '0.875rem'
                            }}>
                              <div style={{ 
                                fontWeight: 'bold', 
                                marginBottom: '0.75rem', 
                                color: '#0066cc',
                                fontSize: '1rem',
                                borderBottom: '2px solid #0066cc',
                                paddingBottom: '0.5rem'
                              }}>
                                Sample {idx + 1}
                              </div>
                              
                              {/* Column Selection Buttons */}
                              <div style={{ 
                                marginBottom: '0.75rem',
                                display: 'flex',
                                flexWrap: 'wrap',
                                gap: '0.5rem'
                              }}>
                                {columnNames.map(colName => (
                                  <Button
                                    key={colName}
                                    variant={selectedColumn === colName ? 'primary' : 'tertiary'}
                                    size="sm"
                                    onClick={() => setSelectedColumnsPerSample(prev => ({
                                      ...prev,
                                      [idx]: colName
                                    }))}
                                    style={{
                                      fontSize: '0.75rem',
                                      padding: '4px 10px',
                                      borderRadius: '16px',
                                      ...(selectedColumn === colName ? {
                                        backgroundColor: '#0066cc',
                                        color: 'white',
                                      } : {
                                        backgroundColor: '#f0f0f0',
                                        color: '#333',
                                        border: '1px solid #d2d2d2',
                                      })
                                    }}
                                  >
                                    {colName}
                                  </Button>
                                ))}
                              </div>
                              
                              {/* Selected Column Value */}
                              {selectedColumn && sample[selectedColumn] !== undefined && (
                                <div style={{
                                  backgroundColor: '#f8f8f8',
                                  borderRadius: '4px',
                                  border: '1px solid #e0e0e0',
                                  overflow: 'hidden'
                                }}>
                                  <div style={{
                                    backgroundColor: '#e8e8e8',
                                    padding: '6px 12px',
                                    fontWeight: 'bold',
                                    fontSize: '0.8rem',
                                    color: '#555',
                                    borderBottom: '1px solid #d0d0d0'
                                  }}>
                                    {selectedColumn}
                                  </div>
                                  <div style={{
                                    padding: '12px',
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    maxHeight: '200px',
                                    overflowY: 'auto',
                                    fontFamily: typeof sample[selectedColumn] === 'string' ? 'inherit' : 'monospace',
                                    fontSize: '0.85rem',
                                    lineHeight: '1.5'
                                  }}>
                                    {typeof sample[selectedColumn] === 'object' 
                                      ? JSON.stringify(sample[selectedColumn], null, 2)
                                      : String(sample[selectedColumn])
                                    }
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        });
                      })()}
                    </div>
                  )}
                </div>
              )}
            </Form>
          </CardBody>
        </Card>
      </GridItem>

      {/* Right Panel - Schema Requirements */}
      <GridItem span={5} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Dataset Requirements
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            {schema && (
              <>
                {/* Show column count */}
                <DescriptionList isHorizontal>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Columns</DescriptionListTerm>
                    <DescriptionListDescription>
                      {(() => {
                        const required = (schema.required_columns || schema.requirements?.required_columns || []).length;
                        const optional = (schema.optional_columns || schema.requirements?.optional_columns || []).length;
                        return required + optional;
                      })()}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>

                {/* Required Columns - Handle both schema formats */}
                {(schema.required_columns || schema.requirements?.required_columns) && (
                  <>
                    <Title headingLevel="h4" size="md" style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
                      Required Columns
                    </Title>
                    <List isPlain isBordered>
                      {(schema.required_columns || schema.requirements?.required_columns || []).map((col) => (
                        <ListItem key={col}>
                          <code>{col}</code>
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}

                {/* Optional Columns */}
                {((schema.optional_columns && schema.optional_columns.length > 0) || 
                  (schema.requirements?.optional_columns && schema.requirements.optional_columns.length > 0)) && (
                      <>
                        <Title headingLevel="h4" size="md" style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
                          Optional Columns
                        </Title>
                        <List isPlain isBordered>
                      {(schema.optional_columns || schema.requirements?.optional_columns || []).map((col) => (
                            <ListItem key={col}>
                              <code>{col}</code>
                            </ListItem>
                          ))}
                        </List>
                      </>
                    )}

                {/* Description */}
                {(schema.description || schema.requirements?.description) && (
                      <>
                        <Title headingLevel="h4" size="md" style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
                          Description
                        </Title>
                        <div style={{ fontSize: '0.875rem' }}>
                      {schema.description || schema.requirements?.description}
                        </div>
                      </>
                    )}

                {/* Minimum samples alert */}
                {(schema.min_samples || schema.requirements?.min_samples) && (
                      <Alert
                        variant={AlertVariant.info}
                        isInline
                        title="Minimum samples required"
                        style={{ marginTop: '1rem' }}
                      >
                    This flow requires at least {schema.min_samples || schema.requirements?.min_samples} samples.
                      </Alert>
                )}

                <div style={{ marginTop: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
                  <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
                    Example Dataset Format
                  </Title>
                  <CodeBlock>
                    <CodeBlockCode>
{`{
  "document": "Your text here...",
  "domain": "Category",
  "icl_document": "Example...",
  ...
}`}
                    </CodeBlockCode>
                  </CodeBlock>
                </div>
              </>
            )}
          </CardBody>
        </Card>
      </GridItem>
    </Grid>

    {/* Duplicate Rows Modal */}
    <DuplicatesModal
      showDuplicatesModal={showDuplicatesModal}
      duplicateInfo={duplicateInfo}
      isRemovingDuplicates={isRemovingDuplicates}
      onRemoveDuplicates={handleRemoveDuplicates}
      onKeepDuplicates={handleKeepDuplicates}
    />

    {/* Missing Columns Modals */}
    <MissingColumnsModal
      missingColumns={missingColumns}
      showMissingColumnsModal={showMissingColumnsModal}
      addingMissingColumns={addingMissingColumns}
      currentMissingColumnIndex={currentMissingColumnIndex}
      missingColumnValues={missingColumnValues}
      currentColumnInput={currentColumnInput}
      numSamples={numSamples}
      onUseRepetitiveFormat={handleUseRepetitiveFormat}
      onCancelAndFixManually={handleCancelAndFixManually}
      onSaveColumnValue={handleSaveMissingColumnValue}
      onCancelAdding={() => dispatchMissingCols({ type: 'CANCEL_ADDING' })}
      onColumnInputChange={(value) => dispatchMissingCols({ type: 'SET_COLUMN_INPUT', payload: value })}
    />

    {/* Unsupported File Format Modal */}
    <Modal
      variant={ModalVariant.small}
      title="Unsupported File Format"
      titleIconVariant="danger"
      isOpen={showUnsupportedFormatError}
      onClose={() => dispatchUnsupportedFormat({ type: 'HIDE' })}
      actions={[
        <Button
          key="ok"
          variant="primary"
          onClick={() => dispatchUnsupportedFormat({ type: 'HIDE' })}
        >
          OK, I'll upload a supported file
        </Button>
      ]}
    >
      <Alert
        variant={AlertVariant.danger}
        isInline
        title="File format not supported"
        style={{ marginBottom: '1.5rem' }}
      >
        <p style={{ marginTop: '0.5rem' }}>
          The file <strong>"{unsupportedFileName}"</strong> has an unsupported format.
        </p>
      </Alert>

      <div style={{ marginTop: '1rem' }}>
        <p style={{ marginBottom: '1rem' }}>
          <strong>Please upload a dataset in one of these supported formats:</strong>
        </p>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: '0.75rem',
          padding: '1rem',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircleIcon style={{ color: '#3e8635' }} />
            <code style={{ backgroundColor: '#e7f5e7', padding: '2px 8px', borderRadius: '4px' }}>.jsonl</code>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircleIcon style={{ color: '#3e8635' }} />
            <code style={{ backgroundColor: '#e7f5e7', padding: '2px 8px', borderRadius: '4px' }}>.json</code>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircleIcon style={{ color: '#3e8635' }} />
            <code style={{ backgroundColor: '#e7f5e7', padding: '2px 8px', borderRadius: '4px' }}>.csv</code>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircleIcon style={{ color: '#3e8635' }} />
            <code style={{ backgroundColor: '#e7f5e7', padding: '2px 8px', borderRadius: '4px' }}>.parquet</code>
          </div>
        </div>

        <p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#6a6e73' }}>
          <strong>Tip:</strong> Parquet files offer the fastest loading performance for large datasets.
        </p>
      </div>
    </Modal>
  </>
  );
};

export default DatasetConfigurationStep;

