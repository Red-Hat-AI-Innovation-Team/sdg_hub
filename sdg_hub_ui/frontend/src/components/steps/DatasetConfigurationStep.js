import React, { useState, useEffect } from 'react';
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
} from '@patternfly/react-core';
import { CheckCircleIcon, UploadIcon, EditIcon, ExclamationTriangleIcon } from '@patternfly/react-icons';
import { datasetAPI } from '../../services/api';

/**
 * Dataset Configuration Step Component
 * 
 * Allows users to:
 * - View dataset schema requirements
 * - Load dataset from file
 * - Configure dataset parameters (num_samples, shuffle, seed)
 * - Preview loaded dataset
 */
const DatasetConfigurationStep = ({ selectedFlow, datasetConfig, importedConfig, onConfigChange, onError }) => {
  const [schema, setSchema] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigured, setIsConfigured] = useState(false);

  // Form state
  const [dataFiles, setDataFiles] = useState('');
  const [numSamples, setNumSamples] = useState(2);
  const [shuffle, setShuffle] = useState(true);
  const [seed, setSeed] = useState(42);
  const [split, setSplit] = useState('train');

  // Pre-fill form with existing datasetConfig or imported configuration
  useEffect(() => {
    const configToUse = importedConfig || datasetConfig;
    
    if (configToUse && Object.keys(configToUse).length > 0) {
      if (configToUse.data_files) {
        setDataFiles(configToUse.data_files);
        setUploadedFilePath(configToUse.data_files); // Set the file path for validation
        
        // Set uploaded filename from config
        if (configToUse.uploaded_file) {
          setUploadedFileName(configToUse.uploaded_file);
        } else if (configToUse.data_files) {
          // Extract filename from path if uploaded_file not provided
          const pathParts = configToUse.data_files.split('/');
          setUploadedFileName(pathParts[pathParts.length - 1]);
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

  // File upload state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [isUploadLoading, setIsUploadLoading] = useState(false);
  const [uploadedFilePath, setUploadedFilePath] = useState(''); // Store the uploaded file path

  // UI state
  const [isPreviewExpanded, setIsPreviewExpanded] = useState(false);
  const [selectedColumnsPerSample, setSelectedColumnsPerSample] = useState({}); // Track selected column for each sample
  
  // Missing columns validation
  const [missingColumns, setMissingColumns] = useState([]);
  const [showMissingColumnsModal, setShowMissingColumnsModal] = useState(false);
  const [addingMissingColumns, setAddingMissingColumns] = useState(false);
  const [currentMissingColumnIndex, setCurrentMissingColumnIndex] = useState(0);
  const [missingColumnValues, setMissingColumnValues] = useState({});
  const [currentColumnInput, setCurrentColumnInput] = useState('');

  /**
   * Load dataset schema when flow is selected
   */
  useEffect(() => {
    if (selectedFlow) {
      loadSchema();
    }
  }, [selectedFlow]);

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
  
  // Unsupported format error state
  const [showUnsupportedFormatError, setShowUnsupportedFormatError] = useState(false);
  const [unsupportedFileName, setUnsupportedFileName] = useState('');

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
      setUnsupportedFileName(file.name);
      setShowUnsupportedFormatError(true);
      return;  // Don't proceed with upload
    }
    
    setIsUploadLoading(true);
    try {
      const fileFormat = getFileFormat(file.name);
      const isBinaryFormat = fileFormat === 'parquet';
      
      // For binary files (Parquet), upload directly without reading content
      if (isBinaryFormat) {
        setUploadedFile(file);  // Store file object for binary formats
        setUploadedFileName(file.name);
        
        // Upload directly and let backend handle it
        await validateUploadedFile(null, file, null, fileFormat);
        setIsUploadLoading(false);
        return;
      }
      
      // For text files (JSON, JSONL, CSV), read content
      const reader = new FileReader();
      reader.onload = async (e) => {
        const fileContent = e.target.result;
        setUploadedFile(fileContent);
        setUploadedFileName(file.name);
        
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
        setIsUploadLoading(false);
      };
      reader.onerror = () => {
        onError('Failed to read file');
        setIsUploadLoading(false);
      };
      reader.readAsText(file);
    } catch (error) {
      onError('Error reading file: ' + error.message);
      setIsUploadLoading(false);
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
      setUploadedFilePath(uploadedPath);
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
      setUploadedFile(null);
      setUploadedFileName('');
      setUploadedFilePath('');
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
    setUploadedFile(null);
    setUploadedFileName('');
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

      // Check for missing columns
      const missing = checkMissingColumns(previewData);
      
      if (missing.length > 0) {
        // Dataset is missing required columns - show modal
        setMissingColumns(missing);
        setShowMissingColumnsModal(true);
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

      // Update parent state
      onConfigChange(finalConfig);
      setIsConfigured(true);

    } catch (error) {
      onError('Failed to reload dataset: ' + error.message);
    } finally {
      setIsLoading(false);
    }
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
      try {
        const response = await datasetAPI.loadDataset(loadConfig);
        
        // Get preview of the FILTERED dataset
        const previewData = await datasetAPI.getPreview();
        setPreview(previewData);
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

      // Update parent state
      onConfigChange(finalConfig);
      setIsConfigured(true);

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

  /**
   * Handle user choosing to add missing columns with repetitive format
   */
  const handleUseRepetitiveFormat = () => {
    setShowMissingColumnsModal(false);
    setAddingMissingColumns(true);
    setCurrentMissingColumnIndex(0);
    setMissingColumnValues({});
    setCurrentColumnInput(''); // Clear input field
  };

  /**
   * Handle user canceling and choosing to fix file manually
   */
  const handleCancelAndFixManually = () => {
    setShowMissingColumnsModal(false);
    setMissingColumns([]);
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
    setMissingColumnValues(newValues);
    
    // Clear the input field for the next column
    setCurrentColumnInput('');
    
    // Move to next column or finish
    if (currentMissingColumnIndex < missingColumns.length - 1) {
      setCurrentMissingColumnIndex(currentMissingColumnIndex + 1);
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
      setAddingMissingColumns(false);
      
      // Detect file format from filename
      const uploadedPath = uploadedFilePath || dataFiles;
      const fileFormat = getFileFormat(uploadedFileName || uploadedPath);
      
      // Preview is already set from the initial load attempt
      // Just mark the dataset as configured
      
      const finalConfig = {
        data_files: uploadedPath, // Use stored path
        file_format: fileFormat,
        num_samples: numSamples || null,
        shuffle,
        seed,
        uploaded_file: uploadedFileName,
        added_columns: columnValues // Track which columns were added
      };


      // Update parent state - this should trigger validation
      if (onConfigChange) {
        onConfigChange(finalConfig);
      }
      setIsConfigured(true);
      
      // Reset missing columns state
      setMissingColumns([]);
      setMissingColumnValues({});
      setCurrentMissingColumnIndex(0);
      setCurrentColumnInput('');

    } catch (error) {
      onError('Failed to apply missing columns: ' + error.message);
    }
  };

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
                        {uploadedFile ? 'File uploaded: ' : 'Referenced file: '}{uploadedFileName}
                      </div>
                      {!uploadedFile && uploadedFilePath && (
                        <div style={{ fontSize: '0.9rem', color: '#3e8635', marginBottom: '0.5rem' }}>
                          ✓ Dataset loaded from configuration
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
                    label="Number of Samples" 
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
                    <Checkbox
                      id="shuffle"
                      label="Shuffle dataset"
                      isChecked={shuffle}
                      onChange={(event, checked) => setShuffle(checked)}
                    />
                  </FormGroup>
                </GridItem>

                {/* Seed (only shown if shuffle is enabled) */}
                {shuffle && (
                  <GridItem span={6}>
                    <FormGroup label="Random Seed" fieldId="seed">
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
                    title="File uploaded - adjust parameters above, then click Load Dataset"
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

    {/* Missing Columns Modal */}
    <Modal
      variant={ModalVariant.medium}
      title="Missing Required Columns"
      isOpen={showMissingColumnsModal}
      onClose={handleCancelAndFixManually}
      actions={[
        <Button
          key="use-repetitive"
          variant="primary"
          onClick={handleUseRepetitiveFormat}
        >
          Use Repetitive Format
        </Button>,
        <Button
          key="cancel"
          variant="secondary"
          onClick={handleCancelAndFixManually}
        >
          Cancel
        </Button>
      ]}
    >
      <Alert
        variant={AlertVariant.warning}
        isInline
        title="Your dataset does not contain all of the required columns"
        style={{ marginBottom: '1.5rem' }}
      >
        <p style={{ marginTop: '0.5rem' }}>
          The following columns are missing from your dataset:
        </p>
        <List isPlain style={{ marginTop: '0.75rem', marginLeft: '1rem' }}>
          {missingColumns.map(col => (
            <ListItem key={col}>
              <code style={{ 
                backgroundColor: '#fff3cd',
                padding: '2px 6px',
                borderRadius: '3px',
                color: '#856404'
              }}>
                {col}
              </code>
            </ListItem>
          ))}
        </List>
      </Alert>

      <div style={{ marginTop: '1.5rem' }}>
        <p style={{ marginBottom: '1rem' }}>
          <strong>You have two options:</strong>
        </p>
        
        <div style={{ 
          padding: '1rem',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px',
          marginBottom: '1rem'
        }}>
          <p style={{ marginBottom: '0.5rem' }}>
            <strong>1. Use Repetitive Format</strong>
          </p>
          <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            Add the same content to all rows in your dataset for each missing column. 
            You'll be guided step by step to fill in each missing column.
          </p>
        </div>
        
        <div style={{ 
          padding: '1rem',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px'
        }}>
          <p style={{ marginBottom: '0.5rem' }}>
            <strong>2. Cancel and Work Manually</strong>
          </p>
          <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            Fix your dataset file manually by adding the missing columns, 
            then upload it again.
          </p>
        </div>
      </div>
    </Modal>

    {/* Add Missing Columns Step-by-Step Modal */}
    <Modal
      variant={ModalVariant.medium}
      title={`Add Missing Column: ${missingColumns[currentMissingColumnIndex]}`}
      isOpen={addingMissingColumns}
      onClose={() => setAddingMissingColumns(false)}
      actions={[
        <Button
          key="save"
          variant="primary"
          onClick={() => {
            const columnName = missingColumns[currentMissingColumnIndex];
            handleSaveMissingColumnValue(columnName, currentColumnInput);
          }}
        >
          {currentMissingColumnIndex < missingColumns.length - 1 ? 'Next Column' : 'Finish & Load Dataset'}
        </Button>,
        <Button
          key="cancel"
          variant="secondary"
          onClick={() => {
            setAddingMissingColumns(false);
            setMissingColumnValues({});
            setCurrentMissingColumnIndex(0);
            setCurrentColumnInput('');
          }}
        >
          Cancel
        </Button>
      ]}
    >
      <Alert
        variant={AlertVariant.info}
        isInline
        title={`Column ${currentMissingColumnIndex + 1} of ${missingColumns.length}`}
        style={{ marginBottom: '1.5rem' }}
      >
        <p>
          This value will be added to <strong>all {numSamples} samples</strong> in your dataset.
        </p>
      </Alert>

      <Form>
        <FormGroup
          label={`Value for "${missingColumns[currentMissingColumnIndex]}"`}
          isRequired
          fieldId="missing-column-value"
          helperText="Enter the value that will be used for all rows in this column"
        >
          <TextInput
            isRequired
            type="text"
            id="missing-column-value"
            name="missing-column-value"
            value={currentColumnInput}
            onChange={(event, value) => setCurrentColumnInput(value)}
            placeholder={`Enter value for ${missingColumns[currentMissingColumnIndex]}`}
          />
        </FormGroup>
      </Form>

      {Object.keys(missingColumnValues).length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
            Previously Added Columns
          </Title>
          <List isPlain isBordered>
            {Object.entries(missingColumnValues).map(([col, val]) => (
              <ListItem key={col}>
                <code>{col}</code>: <strong>{val}</strong>
              </ListItem>
            ))}
          </List>
        </div>
      )}
    </Modal>

    {/* Unsupported File Format Modal */}
    <Modal
      variant={ModalVariant.small}
      title="Unsupported File Format"
      titleIconVariant="danger"
      isOpen={showUnsupportedFormatError}
      onClose={() => setShowUnsupportedFormatError(false)}
      actions={[
        <Button
          key="ok"
          variant="primary"
          onClick={() => setShowUnsupportedFormatError(false)}
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

