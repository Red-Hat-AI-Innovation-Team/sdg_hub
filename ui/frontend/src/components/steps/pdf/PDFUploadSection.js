import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Button,
  Alert,
  AlertVariant,
  Spinner,
  Badge,
  List,
  ListItem,
  Popover,
} from '@patternfly/react-core';
import {
  UploadIcon,
  CheckCircleIcon,
  FileIcon,
  OutlinedQuestionCircleIcon,
  TrashIcon,
  DownloadIcon,
  EyeIcon,
} from '@patternfly/react-icons';
import { Flex, FlexItem } from '@patternfly/react-core';

/**
 * Step 1: Upload & Convert PDF Files
 */
const PDFUploadSection = ({
  isStep1Complete,
  expandedSteps,
  toggleStep,
  uploadedFiles,
  isUploading,
  uploadError,
  convertedFiles,
  conversionProgress,
  conversionError,
  filesBeingConverted,
  jobStatus,
  handleFileUpload,
  handleConvert,
  handleReset,
  handleOpenComparison,
  handleDeleteFile,
  handleDownloadMarkdown,
  setJobStatus,
  setConvertedFiles,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step1')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {isStep1Complete ? (
              <CheckCircleIcon color="var(--pf-global--success-color--100)" />
            ) : (
              <span style={{ 
                width: '24px', 
                height: '24px', 
                borderRadius: '50%', 
                backgroundColor: '#0066cc', 
                color: 'white', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                fontSize: '0.875rem',
                fontWeight: 'bold'
              }}>1</span>
            )}
            Upload & Convert PDF Files
            {isStep1Complete && !expandedSteps.step1 && (
              <Badge isRead style={{ marginLeft: '0.5rem' }}>
                {convertedFiles.length} file{convertedFiles.length !== 1 ? 's' : ''} converted
              </Badge>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Popover
              headerContent="PDF Conversion"
              bodyContent="Upload PDF or Markdown files. PDF files will be converted to Markdown using docling. MD files are ready for chunking immediately."
            >
              <Button variant="plain" aria-label="More info" onClick={(e) => e.stopPropagation()}>
                <OutlinedQuestionCircleIcon />
              </Button>
            </Popover>
            <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
              {expandedSteps.step1 ? '▼' : '▶'}
            </span>
          </div>
        </div>
      </CardTitle>
      {expandedSteps.step1 && <CardBody>
        {/* Upload Area - always visible */}
        <div style={{ 
          border: '2px dashed #d2d2d2', 
          borderRadius: '8px', 
          padding: '1.5rem', 
          textAlign: 'center',
          backgroundColor: '#f5f5f5',
          marginBottom: uploadedFiles.length > 0 ? '1.5rem' : '0'
        }}>
          <UploadIcon size="lg" style={{ marginBottom: '0.75rem', color: '#6a6e73' }} />
          <p style={{ marginBottom: '0.75rem', fontSize: '0.9rem' }}>
            {uploadedFiles.length === 0 
              ? 'Drag and drop PDF/MD files here, or click to browse' 
              : 'Add more PDF/MD files'}
          </p>
          <input
            type="file"
            accept=".pdf,.md"
            multiple
            style={{ display: 'none' }}
            id="pdf-file-input"
            onChange={(e) => handleFileUpload(e, uploadedFiles.length > 0)}
          />
          <Button 
            variant={uploadedFiles.length === 0 ? 'primary' : 'secondary'}
            onClick={() => document.getElementById('pdf-file-input').click()}
            isLoading={isUploading}
            isDisabled={jobStatus === 'converting'}
          >
            {uploadedFiles.length === 0 ? 'Browse Files' : 'Add More Files'}
          </Button>
          
          {uploadError && (
            <Alert variant={AlertVariant.danger} isInline title="Upload Error" style={{ marginTop: '1rem' }}>
              {uploadError}
            </Alert>
          )}
        </div>

        {/* File List with Actions */}
        {uploadedFiles.length > 0 && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <Title headingLevel="h4" size="md">
                Files ({uploadedFiles.length})
              </Title>
              <Button 
                variant="link" 
                isDanger
                onClick={handleReset}
                isDisabled={jobStatus === 'converting' || jobStatus === 'creating_dataset'}
                isSmall
              >
                Clear All
              </Button>
            </div>
            
            <List isPlain>
              {uploadedFiles.map((file, index) => {
                const fileName = file.name || file.filename;
                const isMarkdownFile = fileName.toLowerCase().endsWith('.md');
                const isPdfFile = fileName.toLowerCase().endsWith('.pdf');
                const isConverted = convertedFiles.some(cf => 
                  (typeof cf === 'string' ? cf : cf.original) === fileName ||
                  (typeof cf === 'object' && cf.markdown?.includes(fileName.replace('.pdf', '')))
                );
                const isConverting = filesBeingConverted.has(fileName);
                const convertedFile = convertedFiles.find(cf => 
                  (typeof cf === 'object' && cf.original === fileName) ||
                  (typeof cf === 'string' && cf.includes(fileName.replace('.pdf', '')))
                );
                
                // Determine status
                let status = 'pending';
                let statusColor = '#6a6e73';
                let statusBg = '#f0f0f0';
                let statusText = 'Pending';
                
                if (isMarkdownFile) {
                  status = 'markdown';
                  statusColor = '#0066cc';
                  statusBg = '#e7f1fa';
                  statusText = 'Ready';
                } else if (isConverting) {
                  status = 'converting';
                  statusColor = '#0066cc';
                  statusBg = '#e7f1fa';
                  statusText = 'Converting...';
                } else if (isConverted) {
                  status = 'converted';
                  statusColor = '#3e8635';
                  statusBg = '#f0fff0';
                  statusText = 'Converted';
                }
                
                return (
                  <ListItem 
                    key={fileName || index}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '0.75rem 1rem',
                      backgroundColor: status === 'converted' ? '#f0fff0' : (status === 'markdown' ? '#e7f1fa' : (status === 'converting' ? '#e7f1fa' : (index % 2 === 0 ? '#fafafa' : 'white'))),
                      borderRadius: '6px',
                      marginBottom: '0.5rem',
                      border: status === 'converted' ? '1px solid #3e8635' : (status === 'markdown' ? '1px solid #0066cc' : (status === 'converting' ? '1px solid #0066cc' : '1px solid #e0e0e0')),
                    }}
                  >
                    {/* Left: File Info */}
                    <Flex alignItems={{ default: 'alignItemsCenter' }} style={{ flex: 1, minWidth: 0 }}>
                      <FlexItem>
                        <FileIcon style={{ marginRight: '0.5rem', color: status === 'converted' ? '#3e8635' : (status === 'markdown' ? '#0066cc' : '#c9190b') }} />
                      </FlexItem>
                      <FlexItem style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ 
                          fontWeight: status === 'converted' ? '600' : 'normal',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {fileName}
                        </div>
                        {file.size && (
                          <div style={{ color: '#6a6e73', fontSize: '0.8rem' }}>
                            {(file.size / 1024).toFixed(1)} KB
                          </div>
                        )}
                      </FlexItem>
                    </Flex>
                    
                    {/* Middle: Status Badge */}
                    <Flex style={{ margin: '0 1rem' }}>
                      <Badge style={{ 
                        backgroundColor: statusBg, 
                        color: statusColor,
                        border: `1px solid ${statusColor}`,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem'
                      }}>
                        {status === 'converting' && <Spinner size="sm" />}
                        {(status === 'converted' || status === 'markdown') && <CheckCircleIcon />}
                        {statusText}
                      </Badge>
                    </Flex>
                    
                    {/* Right: Action Buttons */}
                    <Flex style={{ gap: '0.5rem', flexShrink: 0 }}>
                      {/* Convert button - for pending PDF files only */}
                      {status === 'pending' && isPdfFile && (
                        <Button
                          variant="primary"
                          isSmall
                          onClick={() => {
                            setJobStatus('uploaded');
                            handleConvert([fileName]);
                          }}
                          isDisabled={jobStatus === 'converting'}
                        >
                          Convert
                        </Button>
                      )}
                      
                      {/* Reconvert button - for converted PDF files only */}
                      {status === 'converted' && isPdfFile && (
                        <Button
                          variant="secondary"
                          isSmall
                          onClick={() => {
                            setConvertedFiles(prev => prev.filter(cf => {
                              const originalName = typeof cf === 'string' ? cf : cf.original;
                              return originalName !== fileName;
                            }));
                            setJobStatus('uploaded');
                            handleConvert([fileName]);
                          }}
                          isDisabled={jobStatus === 'converting'}
                        >
                          Reconvert
                        </Button>
                      )}
                      
                      {/* Compare button - for converted PDF files */}
                      {status === 'converted' && convertedFile && isPdfFile && (
                        <Button
                          variant="tertiary"
                          isSmall
                          onClick={() => {
                            const markdownFilename = typeof convertedFile === 'string' 
                              ? convertedFile.split('/').pop() 
                              : (convertedFile.markdown?.split('/').pop() || convertedFile.markdown);
                            if (markdownFilename) {
                              handleOpenComparison(fileName, markdownFilename);
                            }
                          }}
                        >
                          <EyeIcon style={{ marginRight: '0.25rem' }} /> Compare
                        </Button>
                      )}
                      
                      {/* Preview button - for uploaded MD files */}
                      {status === 'markdown' && convertedFile && (
                        <Button
                          variant="tertiary"
                          isSmall
                          onClick={() => {
                            const markdownFilename = convertedFile.markdown || fileName;
                            if (markdownFilename) {
                              handleOpenComparison(null, markdownFilename);
                            }
                          }}
                        >
                          <EyeIcon style={{ marginRight: '0.25rem' }} /> Preview
                        </Button>
                      )}
                      
                      {/* Download button - for converted files and MD files */}
                      {(status === 'converted' || status === 'markdown') && convertedFile && (
                        <Button
                          variant="link"
                          isSmall
                          onClick={() => {
                            const markdownFilename = typeof convertedFile === 'string' 
                              ? convertedFile.split('/').pop() 
                              : (convertedFile.markdown?.split('/').pop() || convertedFile.markdown);
                            handleDownloadMarkdown(markdownFilename);
                          }}
                        >
                          <DownloadIcon style={{ marginRight: '0.25rem' }} /> Download
                        </Button>
                      )}
                      
                      {/* Delete button */}
                      <Button
                        variant="plain"
                        aria-label={`Delete ${fileName}`}
                        onClick={() => handleDeleteFile(file)}
                        isDisabled={jobStatus === 'converting' || jobStatus === 'creating_dataset'}
                      >
                        <TrashIcon style={{ color: '#c9190b' }} />
                      </Button>
                    </Flex>
                  </ListItem>
                );
              })}
            </List>
            
            {/* Convert All Pending Button */}
            {(() => {
              const pendingPdfFiles = uploadedFiles.filter(f => {
                const fileName = f.name || f.filename;
                if (!fileName.toLowerCase().endsWith('.pdf')) return false;
                return !convertedFiles.some(cf => 
                  (typeof cf === 'string' ? cf : cf.original) === fileName ||
                  (typeof cf === 'object' && cf.markdown?.includes(fileName.replace('.pdf', '')))
                );
              });
              
              if (pendingPdfFiles.length > 0 && jobStatus !== 'converting') {
                return (
                  <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <Button 
                      variant="primary" 
                      onClick={() => {
                        const pendingFileNames = pendingPdfFiles.map(f => f.name || f.filename);
                        setJobStatus('uploaded');
                        handleConvert(pendingFileNames);
                      }}
                    >
                      Convert All PDFs ({pendingPdfFiles.length} file{pendingPdfFiles.length !== 1 ? 's' : ''})
                    </Button>
                  </div>
                );
              }
              return null;
            })()}
            
            {/* Conversion Error */}
            {conversionError && (
              <Alert variant={AlertVariant.danger} isInline title="Conversion Error" style={{ marginTop: '1rem' }}>
                {conversionError}
              </Alert>
            )}
            
            {/* Conversion Progress Message */}
            {jobStatus === 'converting' && (
              <Alert variant={AlertVariant.info} isInline title="Converting..." style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Spinner size="sm" />
                  <span>
                    {conversionProgress.message || `Converting ${conversionProgress.current + 1} of ${conversionProgress.total} file(s)...`}
                  </span>
                </div>
              </Alert>
            )}
            
            {/* Success Message when all files are ready */}
            {convertedFiles.length > 0 && convertedFiles.length >= uploadedFiles.length && jobStatus !== 'converting' && (() => {
              const mdFilesCount = convertedFiles.filter(cf => cf.isMarkdownUpload).length;
              const convertedPdfsCount = convertedFiles.length - mdFilesCount;
              let message = '';
              if (mdFilesCount > 0 && convertedPdfsCount > 0) {
                message = `${convertedPdfsCount} PDF(s) converted to Markdown and ${mdFilesCount} Markdown file(s) ready. Proceed to chunking below.`;
              } else if (mdFilesCount > 0) {
                message = `All ${mdFilesCount} Markdown file(s) are ready. Proceed to chunking below.`;
              } else {
                message = `All ${convertedPdfsCount} file(s) have been converted to Markdown. Proceed to chunking below.`;
              }
              return (
                <Alert variant={AlertVariant.success} isInline title="All files ready" style={{ marginTop: '1rem' }}>
                  {message}
                </Alert>
              );
            })()}
          </div>
        )}
      </CardBody>}
    </Card>
  );
};

export default PDFUploadSection;
