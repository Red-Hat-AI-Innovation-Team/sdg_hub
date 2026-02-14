import React from 'react';
import {
  Card,
  CardBody,
  Title,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  List,
  ListItem,
  Modal,
  ModalVariant,
  FormGroup,
  TextInput,
  Tooltip,
  Badge,
} from '@patternfly/react-core';
import { 
  CheckCircleIcon, 
  DownloadIcon,
  OutlinedQuestionCircleIcon,
} from '@patternfly/react-icons';
import { preprocessingAPI } from '../../services/api';
import { usePDFProcessing } from '../../hooks/usePDFProcessing';
import { useExecutionConfig } from '../../contexts/ExecutionContext';

// Sub-components
import PDFUploadSection from './pdf/PDFUploadSection';
import ChunkingSettings from './pdf/ChunkingSettings';
import BasicInfoSection from './pdf/BasicInfoSection';
import ICLTemplateSection from './pdf/ICLTemplateSection';
import DatasetPreviewSection from './pdf/DatasetPreviewSection';
import AdditionalColumnsSection from './pdf/AdditionalColumnsSection';

/**
 * SDG Term definitions for tooltips
 */
const SDG_TERM_DEFINITIONS = {
  'icl_template': {
    term: 'ICL Template',
    definition: 'In-Context Learning (ICL) template provides example question-answer pairs that help the model understand the expected format and style of outputs. Good ICL examples improve generation quality.',
  },
  'icl_document': {
    term: 'ICL Document',
    definition: 'A sample document used in the ICL template to show the model what kind of source material it will be working with.',
  },
  'domain': {
    term: 'Domain',
    definition: 'The subject area or category of your documents (e.g., "Medical Research", "Legal Contracts", "Technical Documentation"). This helps the model generate contextually appropriate content.',
  },
  'document_outline': {
    term: 'Document Outline',
    definition: 'A high-level description of your document\'s structure and main topics. This helps the model understand the document organization when generating questions and answers.',
  },
  'chunk_size': {
    term: 'Chunk Size',
    definition: 'The number of words per chunk when splitting documents. Larger chunks provide more context but may exceed model limits. Recommended: 500-1500 words.',
  },
  'chunk_overlap': {
    term: 'Chunk Overlap',
    definition: 'The number of words that overlap between consecutive chunks. Overlap helps maintain context at chunk boundaries. Recommended: 10-20% of chunk size.',
  },
};

/**
 * Tooltip component for SDG terms
 */
const SDGTermTooltip = ({ term, children }) => {
  const termInfo = SDG_TERM_DEFINITIONS[term];
  if (!termInfo) return children;
  
  return (
    <Tooltip
      content={
        <div style={{ maxWidth: '300px' }}>
          <strong>{termInfo.term}</strong>
          <p style={{ marginTop: '0.5rem' }}>{termInfo.definition}</p>
        </div>
      }
    >
      <span style={{ 
        borderBottom: '1px dashed #6a6e73', 
        cursor: 'help',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.25rem',
      }}>
        {children}
        <OutlinedQuestionCircleIcon style={{ fontSize: '0.875rem', color: '#6a6e73' }} />
      </span>
    </Tooltip>
  );
};

/**
 * PDF Preprocessing Step Component
 * 
 * Allows users to:
 * - Upload PDF/MD files
 * - Convert them to Markdown using docling
 * - Chunk the documents with configurable parameters
 * - Add required columns (ICL templates, domain, etc.)
 * - Create a final dataset for flow execution
 * 
 * State is managed by the usePDFProcessing custom hook (useReducer).
 */
const PDFPreprocessingStep = ({ 
  selectedFlow: selectedFlowProp, 
  requiredColumns = [],
  onDatasetCreated,
  onCancel,
  onError,
  savedState = null,  // Optional: restored state from parent
  onStateChange = null // Optional: callback to notify parent of state changes
}) => {
  // Use ExecutionContext for selectedFlow (eliminates prop drilling), with prop fallback
  const { selectedFlow: selectedFlowFromContext } = useExecutionConfig();
  const selectedFlow = selectedFlowFromContext || selectedFlowProp;

  // All state, setters, handlers, and core computed values come from the custom hook
  const {
    // State values
    jobId, jobStatus, uploadedFiles, isUploading, uploadError,
    conversionProgress, convertedFiles, conversionError, selectedFilesForConversion, filesBeingConverted,
    chunkSize, chunkOverlap, chunks, totalChunks, isChunking,
    fileChunkConfigs, selectedFilesForChunking, expandedChunkPreviews,
    domain, documentOutline, additionalColumns, fileBasicInfo, editingBasicInfo,
    chunkModalOpen, selectedChunkForModal, comparisonModalOpen, comparisonFile, comparisonMarkdownContent, isLoadingMarkdown,
    datasetNameModalOpen, datasetName, pendingAction,
    datasetCreationSnapshot, hasChangesAfterCreation,
    expandedSteps,
    previewSelectedColumns, previewSelectedFiles, previewSamplesPerFile, previewFileSearchQuery, previewFileSearchFocused, previewExpandedFiles, previewAutoExpanded,
    iclTemplates, selectedTemplateIndex, useCustomICL, customICL,
    fileICLConfigs, editingFileICL, expandedICLFiles, fileAllChunks, selectedChunkIdx, loadingChunks,
    activeTabKey, showChunksPreview, isCreatingDataset,
    createdDatasetInfo,

    // Setter functions
    setJobId, setJobStatus, setUploadedFiles,
    setConversionProgress, setConvertedFiles, setConversionError, setSelectedFilesForConversion, setFilesBeingConverted,
    setChunkSize, setChunkOverlap, setChunks, setTotalChunks,
    setFileChunkConfigs, setSelectedFilesForChunking, setExpandedChunkPreviews,
    setDomain, setDocumentOutline, setAdditionalColumns, setFileBasicInfo, setEditingBasicInfo,
    setChunkModalOpen, setSelectedChunkForModal, setComparisonModalOpen, setComparisonFile, setComparisonMarkdownContent,
    setDatasetNameModalOpen, setDatasetName, setPendingAction,
    setPreviewSelectedColumns, setPreviewSelectedFiles, setPreviewSamplesPerFile, setPreviewFileSearchQuery, setPreviewFileSearchFocused, setPreviewExpandedFiles, setPreviewAutoExpanded,
    setSelectedTemplateIndex, setUseCustomICL, setCustomICL,
    setFileICLConfigs, setEditingFileICL, setExpandedICLFiles,
    setExpandedSteps,

    // Handler functions
    handleFileUpload, handleConvert, handleChunk, loadChunkForFile,
    handleCreateDataset, handleDownloadDataset, handleReset,
    handleOpenComparison, handleDeleteFile, handleDownloadMarkdown,
    canCreateDataset,

    // Computed values from hook
    needsICL, needsDomain, needsDocumentOutline, contentColumnName, needsBasicInfoStep,
  } = usePDFProcessing({
    selectedFlow, requiredColumns, onDatasetCreated, onCancel, onError, savedState, onStateChange,
  });

  // Get non-document required columns that need manual input
  const manualInputColumns = requiredColumns.filter(col => 
    !col.startsWith('icl_') && col !== 'document' && col !== 'text' && col !== 'document_outline' && col !== 'domain'
  );

  // Toggle step expansion
  const toggleStep = (step) => {
    setExpandedSteps(prev => ({ ...prev, [step]: !prev[step] }));
  };

  // Check completion status for each step
  const isStep1Complete = uploadedFiles.length > 0 && convertedFiles.length >= uploadedFiles.length;
  const isStep2Complete = Object.values(fileChunkConfigs).length > 0 && 
    convertedFiles.every(cf => {
      const fileName = typeof cf === 'string' ? cf : cf.original;
      return fileChunkConfigs[fileName]?.isChunked;
    });
  const isStep3Complete = needsBasicInfoStep 
    ? (Object.values(fileChunkConfigs).filter(c => c.isChunked).length > 0 &&
        Object.values(fileChunkConfigs).filter(c => c.isChunked).every((_, idx) => {
          const chunkedFiles = Object.keys(fileChunkConfigs).filter(fn => fileChunkConfigs[fn]?.isChunked);
          return fileBasicInfo[chunkedFiles[idx]]?.isComplete;
        }))
    : Object.values(fileChunkConfigs).some(c => c.isChunked);

  const filesReadyForNextStep = needsBasicInfoStep
    ? Object.values(fileBasicInfo).some(info => info.isComplete)
    : Object.values(fileChunkConfigs).some(c => c.isChunked);

  // Count incomplete files from previous steps
  const unconvertedFilesCount = uploadedFiles.length - convertedFiles.length;
  const unchunkedFilesCount = convertedFiles.length - Object.values(fileChunkConfigs).filter(c => c.isChunked).length;
  const filesWithoutBasicInfoCount = needsBasicInfoStep 
    ? (Object.values(fileChunkConfigs).filter(c => c.isChunked).length - 
       Object.values(fileBasicInfo).filter(info => info.isComplete).length)
    : 0;

  return (
    <div style={{ padding: '1rem' }}>
      {onCancel && (
        <div style={{ marginBottom: '0.5rem' }}>
          <Button 
            variant="link" 
            onClick={onCancel}
            style={{ fontSize: '1rem', fontWeight: '500', paddingLeft: 0 }}
          >
            ← Change Dataset Source
          </Button>
        </div>
      )}
      <Title headingLevel="h2" size="lg" style={{ marginBottom: '1.5rem' }}>
        PDF Preprocessing Pipeline
      </Title>
      
      {/* Step 1: Upload & Convert PDFs */}
      <PDFUploadSection
        isStep1Complete={isStep1Complete}
        expandedSteps={expandedSteps}
        toggleStep={toggleStep}
        uploadedFiles={uploadedFiles}
        isUploading={isUploading}
        uploadError={uploadError}
        convertedFiles={convertedFiles}
        conversionProgress={conversionProgress}
        conversionError={conversionError}
        filesBeingConverted={filesBeingConverted}
        jobStatus={jobStatus}
        handleFileUpload={handleFileUpload}
        handleConvert={handleConvert}
        handleReset={handleReset}
        handleOpenComparison={handleOpenComparison}
        handleDeleteFile={handleDeleteFile}
        handleDownloadMarkdown={handleDownloadMarkdown}
        setJobStatus={setJobStatus}
        setConvertedFiles={setConvertedFiles}
      />

      {/* Step 2: Chunking */}
      {convertedFiles.length > 0 && (
        <ChunkingSettings
          isStep2Complete={isStep2Complete}
          expandedSteps={expandedSteps}
          toggleStep={toggleStep}
          convertedFiles={convertedFiles}
          fileChunkConfigs={fileChunkConfigs}
          selectedFilesForChunking={selectedFilesForChunking}
          expandedChunkPreviews={expandedChunkPreviews}
          chunkSize={chunkSize}
          chunkOverlap={chunkOverlap}
          isChunking={isChunking}
          unconvertedFilesCount={unconvertedFilesCount}
          setChunkSize={setChunkSize}
          setChunkOverlap={setChunkOverlap}
          setFileChunkConfigs={setFileChunkConfigs}
          setSelectedFilesForChunking={setSelectedFilesForChunking}
          setExpandedChunkPreviews={setExpandedChunkPreviews}
          handleChunk={handleChunk}
          setSelectedChunkForModal={setSelectedChunkForModal}
          setChunkModalOpen={setChunkModalOpen}
        />
      )}

      {/* Step 3: Basic Info (per-file) */}
      {Object.values(fileChunkConfigs).some(c => c.isChunked) && needsBasicInfoStep && (
        <BasicInfoSection
          isStep3Complete={isStep3Complete}
          expandedSteps={expandedSteps}
          toggleStep={toggleStep}
          fileChunkConfigs={fileChunkConfigs}
          fileBasicInfo={fileBasicInfo}
          editingBasicInfo={editingBasicInfo}
          needsBasicInfoStep={needsBasicInfoStep}
          needsDomain={needsDomain}
          needsDocumentOutline={needsDocumentOutline}
          unchunkedFilesCount={unchunkedFilesCount}
          setFileBasicInfo={setFileBasicInfo}
          setEditingBasicInfo={setEditingBasicInfo}
          setDomain={setDomain}
          setDocumentOutline={setDocumentOutline}
          setJobStatus={setJobStatus}
        />
      )}

      {/* Step 4: ICL Template (per-file) */}
      {filesReadyForNextStep && needsICL && (
        <ICLTemplateSection
          expandedSteps={expandedSteps}
          toggleStep={toggleStep}
          filesReadyForNextStep={filesReadyForNextStep}
          needsICL={needsICL}
          fileICLConfigs={fileICLConfigs}
          editingFileICL={editingFileICL}
          expandedICLFiles={expandedICLFiles}
          fileBasicInfo={fileBasicInfo}
          fileChunkConfigs={fileChunkConfigs}
          iclTemplates={iclTemplates}
          filesWithoutBasicInfoCount={filesWithoutBasicInfoCount}
          selectedChunkIdx={selectedChunkIdx}
          loadingChunks={loadingChunks}
          setEditingFileICL={setEditingFileICL}
          setFileICLConfigs={setFileICLConfigs}
          setExpandedICLFiles={setExpandedICLFiles}
          setUseCustomICL={setUseCustomICL}
          setCustomICL={setCustomICL}
          setSelectedTemplateIndex={setSelectedTemplateIndex}
          loadChunkForFile={loadChunkForFile}
          SDGTermTooltip={SDGTermTooltip}
        />
      )}

      {/* Step 5: Dataset Preview */}
      {(needsICL ? Object.values(fileICLConfigs).some(c => c.isComplete) : filesReadyForNextStep) && (
        <DatasetPreviewSection
          expandedSteps={expandedSteps}
          toggleStep={toggleStep}
          needsICL={needsICL}
          needsBasicInfoStep={needsBasicInfoStep}
          needsDomain={needsDomain}
          needsDocumentOutline={needsDocumentOutline}
          contentColumnName={contentColumnName}
          fileICLConfigs={fileICLConfigs}
          fileBasicInfo={fileBasicInfo}
          fileChunkConfigs={fileChunkConfigs}
          iclTemplates={iclTemplates}
          filesReadyForNextStep={filesReadyForNextStep}
          previewSelectedColumns={previewSelectedColumns}
          previewSelectedFiles={previewSelectedFiles}
          previewSamplesPerFile={previewSamplesPerFile}
          previewFileSearchQuery={previewFileSearchQuery}
          previewFileSearchFocused={previewFileSearchFocused}
          previewExpandedFiles={previewExpandedFiles}
          previewAutoExpanded={previewAutoExpanded}
          setPreviewSelectedColumns={setPreviewSelectedColumns}
          setPreviewSelectedFiles={setPreviewSelectedFiles}
          setPreviewSamplesPerFile={setPreviewSamplesPerFile}
          setPreviewFileSearchQuery={setPreviewFileSearchQuery}
          setPreviewFileSearchFocused={setPreviewFileSearchFocused}
          setPreviewExpandedFiles={setPreviewExpandedFiles}
          setPreviewAutoExpanded={setPreviewAutoExpanded}
        />
      )}

      {/* Step 6: Additional Columns */}
      {filesReadyForNextStep && manualInputColumns.length > 0 && (
        <AdditionalColumnsSection
          expandedSteps={expandedSteps}
          toggleStep={toggleStep}
          manualInputColumns={manualInputColumns}
          additionalColumns={additionalColumns}
          needsICL={needsICL}
          filesWithoutBasicInfoCount={filesWithoutBasicInfoCount}
          setAdditionalColumns={setAdditionalColumns}
        />
      )}

      {/* Action Buttons - show when files are ready for dataset creation */}
      {filesReadyForNextStep && jobStatus !== 'complete' && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'flex-end', 
          marginTop: '1.5rem',
          paddingTop: '1rem',
          borderTop: '1px solid #d2d2d2'
        }}>
          <Button 
            variant="primary" 
            onClick={() => {
              // Generate default name from first file before opening modal
              const firstFile = uploadedFiles[0]?.name || 'dataset';
              const baseName = firstFile.replace(/\.[^/.]+$/, ''); // Remove extension
              setDatasetName(`${baseName}_preprocessed`);
              setPendingAction('create');
              setDatasetNameModalOpen(true);
            }}
            isDisabled={!canCreateDataset()}
            isLoading={isCreatingDataset}
          >
            Create Dataset & Continue
          </Button>
        </div>
      )}
      
      {jobStatus === 'complete' && createdDatasetInfo && (
        <Card style={{ 
          marginTop: '1rem', 
          backgroundColor: hasChangesAfterCreation ? '#fff8e6' : '#f0fff0', 
          border: hasChangesAfterCreation ? '1px solid #f0ab00' : '1px solid #3e8635' 
        }}>
          <CardBody>
            {hasChangesAfterCreation && (
              <Alert 
                variant={AlertVariant.warning} 
                isInline 
                title="Configuration Changed" 
                style={{ marginBottom: '1rem' }}
              >
                You've made changes since creating this dataset. Click "Update Dataset" to apply your changes, or press Next to continue with the current version.
              </Alert>
            )}
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
              <CheckCircleIcon 
                color={hasChangesAfterCreation ? "var(--pf-global--warning-color--100)" : "var(--pf-global--success-color--100)"} 
                style={{ fontSize: '2rem' }} 
              />
              <div>
                <Title headingLevel="h3" size="lg" style={{ color: hasChangesAfterCreation ? '#856404' : '#3e8635', marginBottom: '0.25rem' }}>
                  {hasChangesAfterCreation ? 'Dataset Ready (Changes Pending)' : 'Dataset Created Successfully!'}
                </Title>
                <p style={{ color: '#6a6e73', margin: 0 }}>
                  Your preprocessed dataset with <strong>{createdDatasetInfo.num_records} records</strong> is ready. Press <strong>Next</strong> to continue to dataset configuration.
                </p>
              </div>
            </div>
            
            <div style={{ 
              backgroundColor: 'white', 
              padding: '1rem', 
              borderRadius: '4px',
              marginBottom: '1rem'
            }}>
              <p style={{ marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset Details:</p>
              <List isPlain>
                <ListItem><strong>Name:</strong> {datasetName || 'preprocessed_dataset'}.jsonl</ListItem>
                <ListItem><strong>Records:</strong> {createdDatasetInfo.num_records}</ListItem>
                <ListItem><strong>Columns:</strong> {createdDatasetInfo.columns?.join(', ')}</ListItem>
              </List>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {hasChangesAfterCreation && (
                <Button 
                  variant="warning" 
                  onClick={handleCreateDataset}
                  isLoading={isCreatingDataset}
                >
                  Update Dataset
                </Button>
              )}
              <Button 
                variant="secondary" 
                onClick={() => handleDownloadDataset()}
              >
                <DownloadIcon style={{ marginRight: '0.5rem' }} />
                Download Dataset
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Dataset Naming Modal */}
      <Modal
        variant={ModalVariant.small}
        title="Name Your Dataset"
        isOpen={datasetNameModalOpen}
        onClose={() => {
          setDatasetNameModalOpen(false);
          setPendingAction(null);
        }}
        actions={[
          <Button 
            key="confirm" 
            variant="primary" 
            onClick={() => {
              setDatasetNameModalOpen(false);
              setPendingAction(null);
              handleCreateDataset();
            }}
            isDisabled={!datasetName.trim()}
          >
            Create Dataset
          </Button>,
          <Button 
            key="cancel" 
            variant="link" 
            onClick={() => {
              setDatasetNameModalOpen(false);
              setPendingAction(null);
            }}
          >
            Cancel
          </Button>
        ]}
      >
        <div>
          <p style={{ marginBottom: '1rem', color: '#6a6e73' }}>
            Choose a name for your dataset. This name will be used for downloading and will appear in the Dashboard.
          </p>
          <FormGroup
            label="Dataset Name"
            fieldId="dataset-name"
            isRequired
          >
            <TextInput
              id="dataset-name"
              value={datasetName}
              onChange={(_, value) => setDatasetName(value.replace(/[^a-zA-Z0-9_-]/g, '_'))}
              placeholder="e.g., my_preprocessed_data"
            />
            <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginTop: '0.25rem' }}>
              Only letters, numbers, underscores, and hyphens allowed. File will be saved as "{datasetName || 'dataset'}.jsonl"
            </div>
          </FormGroup>
        </div>
      </Modal>

      {/* Chunk Content Modal */}
      <Modal
        variant={ModalVariant.large}
        title={selectedChunkForModal ? `Chunk ${(selectedChunkForModal.chunk?.chunk_index || selectedChunkForModal.index) + 1} - ${selectedChunkForModal.fileName}` : 'Chunk Content'}
        isOpen={chunkModalOpen}
        onClose={() => {
          setChunkModalOpen(false);
          setSelectedChunkForModal(null);
        }}
        actions={[
          <Button key="close" variant="primary" onClick={() => {
            setChunkModalOpen(false);
            setSelectedChunkForModal(null);
          }}>
            Close
          </Button>
        ]}
      >
        {selectedChunkForModal && (
          <div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              marginBottom: '1rem',
              padding: '0.5rem 0.75rem',
              backgroundColor: '#f5f5f5',
              borderRadius: '4px'
            }}>
              <span style={{ fontWeight: 500 }}>
                Chunk {(selectedChunkForModal.chunk?.chunk_index || selectedChunkForModal.index) + 1}
              </span>
              <span style={{ color: '#6a6e73' }}>
                {selectedChunkForModal.chunk?.word_count || selectedChunkForModal.chunk?.document?.split(/\s+/).length || 0} words
              </span>
            </div>
            <div style={{ 
              fontFamily: 'monospace', 
              fontSize: '0.85rem',
              whiteSpace: 'pre-wrap',
              lineHeight: '1.6',
              padding: '1rem',
              backgroundColor: '#fafafa',
              borderRadius: '4px',
              border: '1px solid #e0e0e0',
              maxHeight: '60vh',
              overflowY: 'auto',
            }}>
              {selectedChunkForModal.chunk?.document || 'No content available'}
            </div>
          </div>
        )}
      </Modal>

      {/* PDF vs Markdown Comparison Modal (or Markdown Preview for MD files) */}
      <Modal
        variant={ModalVariant.large}
        title={comparisonFile 
          ? (comparisonFile.pdfFilename 
              ? `Compare: ${comparisonFile.pdfFilename}` 
              : `Preview: ${comparisonFile.markdownFilename}`)
          : 'Markdown Preview'}
        isOpen={comparisonModalOpen}
        onClose={() => {
          setComparisonModalOpen(false);
          setComparisonFile(null);
          setComparisonMarkdownContent('');
        }}
        actions={[
          <Button key="close" variant="primary" onClick={() => {
            setComparisonModalOpen(false);
            setComparisonFile(null);
            setComparisonMarkdownContent('');
          }}>
            Close
          </Button>
        ]}
        style={{ '--pf-v5-c-modal-box--Width': comparisonFile?.pdfFilename ? '95vw' : '70vw', '--pf-v5-c-modal-box--MaxWidth': comparisonFile?.pdfFilename ? '1600px' : '900px' }}
      >
        {comparisonFile && (
          <div style={{ display: 'flex', gap: '1rem', height: '70vh' }}>
            {/* Left side: PDF viewer (only shown for PDF comparisons) */}
            {comparisonFile.pdfFilename && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <div style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#c9190b', 
                  color: 'white', 
                  borderRadius: '4px 4px 0 0',
                  fontWeight: 600,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <span>Original PDF</span>
                  <a 
                    href={preprocessingAPI.getPdfUrl(jobId, comparisonFile.pdfFilename)} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ color: 'white', fontSize: '0.85rem', textDecoration: 'underline' }}
                  >
                    Open in new tab
                  </a>
                </div>
                <div style={{ 
                  flex: 1, 
                  border: '1px solid #d2d2d2', 
                  borderTop: 'none',
                  borderRadius: '0 0 4px 4px',
                  overflow: 'hidden'
                }}>
                  <object
                    data={preprocessingAPI.getPdfUrl(jobId, comparisonFile.pdfFilename)}
                    type="application/pdf"
                    style={{ width: '100%', height: '100%' }}
                    title="PDF Preview"
                  >
                    <div style={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      alignItems: 'center', 
                      justifyContent: 'center', 
                      height: '100%',
                      padding: '2rem',
                      textAlign: 'center',
                      color: '#6a6e73'
                    }}>
                      <p>PDF preview not available in this browser.</p>
                      <a 
                        href={preprocessingAPI.getPdfUrl(jobId, comparisonFile.pdfFilename)} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={{ color: '#0066cc', marginTop: '1rem' }}
                      >
                        Click here to open PDF in a new tab
                      </a>
                    </div>
                  </object>
                </div>
              </div>
            )}
            
            {/* Right side (or full width for MD files): Markdown content */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <div style={{ 
                padding: '0.5rem 1rem', 
                backgroundColor: '#0066cc', 
                color: 'white', 
                borderRadius: '4px 4px 0 0',
                fontWeight: 600 
              }}>
                {comparisonFile.pdfFilename ? 'Converted Markdown' : 'Markdown Content'}
              </div>
              <div style={{ 
                flex: 1, 
                border: '1px solid #d2d2d2', 
                borderTop: 'none',
                borderRadius: '0 0 4px 4px',
                overflow: 'auto',
                padding: '1rem',
                backgroundColor: '#fafafa'
              }}>
                {isLoadingMarkdown ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                    <Spinner size="lg" />
                    <span style={{ marginLeft: '0.5rem' }}>Loading markdown...</span>
                  </div>
                ) : (
                  <pre style={{ 
                    fontFamily: 'monospace', 
                    fontSize: '0.85rem', 
                    whiteSpace: 'pre-wrap', 
                    wordBreak: 'break-word',
                    lineHeight: '1.6',
                    margin: 0
                  }}>
                    {comparisonMarkdownContent}
                  </pre>
                )}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default PDFPreprocessingStep;
