import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Button,
  Alert,
  AlertVariant,
  Badge,
  Grid,
  GridItem,
  NumberInput,
  Popover,
  Tooltip,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  FileIcon,
  OutlinedQuestionCircleIcon,
  CubesIcon,
} from '@patternfly/react-icons';
import { Flex, FlexItem } from '@patternfly/react-core';

/**
 * Step 2: Configure Chunking
 */
const ChunkingSettings = ({
  isStep2Complete,
  expandedSteps,
  toggleStep,
  convertedFiles,
  fileChunkConfigs,
  selectedFilesForChunking,
  expandedChunkPreviews,
  chunkSize,
  chunkOverlap,
  isChunking,
  unconvertedFilesCount,
  setChunkSize,
  setChunkOverlap,
  setFileChunkConfigs,
  setSelectedFilesForChunking,
  setExpandedChunkPreviews,
  handleChunk,
  setSelectedChunkForModal,
  setChunkModalOpen,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step2')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {isStep2Complete ? (
              <CheckCircleIcon color="var(--pf-global--success-color--100)" />
            ) : Object.values(fileChunkConfigs).some(c => c.isChunked) ? (
              <CheckCircleIcon color="var(--pf-global--warning-color--100)" />
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
              }}>2</span>
            )}
            Configure Chunking
            {!expandedSteps.step2 && Object.values(fileChunkConfigs).some(c => c.isChunked) && (
              <Badge isRead style={{ marginLeft: '0.5rem' }}>
                {Object.values(fileChunkConfigs).filter(c => c.isChunked).length} file{Object.values(fileChunkConfigs).filter(c => c.isChunked).length !== 1 ? 's' : ''} chunked
              </Badge>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Popover
              headerContent="Document Chunking"
              bodyContent="Splits documents into smaller chunks for processing. You can configure different chunk sizes for different files."
            >
              <Button variant="plain" aria-label="More info" onClick={(e) => e.stopPropagation()}>
                <OutlinedQuestionCircleIcon />
              </Button>
            </Popover>
            <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
              {expandedSteps.step2 ? '▼' : '▶'}
            </span>
          </div>
        </div>
      </CardTitle>
      {expandedSteps.step2 && <CardBody>
        {/* Chunking Explanation */}
        <Alert variant={AlertVariant.info} isInline title="What is Chunking?" style={{ marginBottom: '1rem' }}>
          <p style={{ marginBottom: '0.5rem' }}>
            <strong>Chunking</strong> splits your documents into smaller, manageable pieces (chunks) that can be processed by language models. 
            This is essential because most models have input length limits.
          </p>
          <p style={{ marginBottom: 0 }}>
            <strong>Tip:</strong> Try different configurations! Use <em>Apply</em> to preview how your document splits, adjust the settings, and compare results. 
            When you're satisfied, click <em>Confirm</em> to finalize.
          </p>
        </Alert>

        {/* Warning about unconverted files */}
        {unconvertedFilesCount > 0 && (
          <Alert variant={AlertVariant.warning} isInline title={`${unconvertedFilesCount} file(s) not converted`} style={{ marginBottom: '1rem' }}>
            Some files have not been converted in Step 1. Go back to Step 1 to convert them before chunking.
          </Alert>
        )}
        
        {/* File Selection for Chunking */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ marginBottom: '0.75rem' }}>
            <Title headingLevel="h4" size="md">
              Click a File to Configure Chunking
            </Title>
          </div>
          
          {convertedFiles.map((cf, index) => {
            const fileName = typeof cf === 'string' ? cf : cf.original;
            const fileConfig = fileChunkConfigs[fileName];
            const isSelected = selectedFilesForChunking.has(fileName);
            const isChunked = fileConfig?.isChunked;
            const isConfirmed = fileConfig?.isConfirmed;
            const needsReApply = fileConfig?.needsReApply;
            const isPreviewExpanded = expandedChunkPreviews.has(fileName);
            const showInlineConfig = (isSelected && selectedFilesForChunking.size === 1) || (isChunked && !isConfirmed);
            
            return (
              <div key={fileName || index} style={{ marginBottom: '0.5rem' }}>
                {/* File Row */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem 1rem',
                    backgroundColor: isSelected ? '#e7f1fa' : (isChunked ? '#f0fff0' : (index % 2 === 0 ? '#fafafa' : 'white')),
                    borderRadius: (isPreviewExpanded || showInlineConfig) ? '6px 6px 0 0' : '6px',
                    border: isSelected ? '2px solid #0066cc' : (isChunked && !isConfirmed ? '2px solid #f0ab00' : (isChunked ? '1px solid #3e8635' : '1px solid #e0e0e0')),
                    borderBottom: (isPreviewExpanded || showInlineConfig) ? 'none' : undefined,
                    cursor: 'pointer',
                    transition: 'background-color 0.15s ease',
                  }}
                  onClick={() => {
                    if (showInlineConfig) {
                      if (isChunked) {
                        setFileChunkConfigs(prev => ({
                          ...prev,
                          [fileName]: { ...prev[fileName], isConfirmed: true, needsReApply: false }
                        }));
                      }
                      setExpandedChunkPreviews(prev => {
                        const newSet = new Set(prev);
                        newSet.delete(fileName);
                        return newSet;
                      });
                      setSelectedFilesForChunking(prev => {
                        const newSet = new Set(prev);
                        newSet.delete(fileName);
                        return newSet;
                      });
                      return;
                    }
                    
                    if (isConfirmed) {
                      setFileChunkConfigs(prev => ({
                        ...prev,
                        [fileName]: { ...prev[fileName], isConfirmed: false, needsReApply: true }
                      }));
                      setExpandedChunkPreviews(prev => {
                        const newSet = new Set(prev);
                        newSet.add(fileName);
                        return newSet;
                      });
                      return;
                    }
                    
                    const newSet = new Set(selectedFilesForChunking);
                    if (isSelected) {
                      newSet.delete(fileName);
                    } else {
                      newSet.clear();
                      newSet.add(fileName);
                    }
                    setSelectedFilesForChunking(newSet);
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.backgroundColor = '#f0f7ff';
                      e.currentTarget.style.borderColor = '#0066cc';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected && !(isChunked && !isConfirmed)) {
                      e.currentTarget.style.backgroundColor = isChunked ? '#f0fff0' : (index % 2 === 0 ? '#fafafa' : 'white');
                      e.currentTarget.style.borderColor = isChunked ? '#3e8635' : '#e0e0e0';
                    }
                  }}
                >
                  {/* Left: File Info */}
                  <Flex alignItems={{ default: 'alignItemsCenter' }} style={{ flex: 1 }}>
                    <FlexItem>
                      <FileIcon style={{ marginRight: '0.5rem', color: isChunked ? '#3e8635' : '#0066cc' }} />
                    </FlexItem>
                    <FlexItem style={{ flex: 1 }}>
                      <span>{fileName}</span>
                      {!isChunked && (
                        <span style={{ fontSize: '0.75rem', color: '#6a6e73', marginLeft: '0.5rem' }}>
                          (click to configure)
                        </span>
                      )}
                      {isChunked && isConfirmed && (
                        <span style={{ fontSize: '0.75rem', color: '#6a6e73', marginLeft: '0.5rem' }}>
                          (click to reconfigure)
                        </span>
                      )}
                    </FlexItem>
                  </Flex>
                  
                  {/* Middle: Chunk Config Status */}
                  <Flex style={{ margin: '0 1rem', gap: '0.5rem' }}>
                    {isChunked ? (
                      <>
                        <Badge style={{ 
                          backgroundColor: isConfirmed ? '#f0fff0' : '#fff3cd', 
                          color: isConfirmed ? '#3e8635' : '#856404', 
                          border: isConfirmed ? '1px solid #3e8635' : '1px solid #f0ab00' 
                        }}>
                          <CheckCircleIcon style={{ marginRight: '0.25rem' }} />
                          {fileConfig.totalChunks} chunks {!isConfirmed && '(pending)'}
                        </Badge>
                        <Badge isRead>
                          Size: {fileConfig.chunkSize} | Overlap: {fileConfig.chunkOverlap}
                        </Badge>
                      </>
                    ) : (
                      <Badge style={{ backgroundColor: '#f0f0f0', color: '#6a6e73' }}>
                        Not chunked
                      </Badge>
                    )}
                  </Flex>
                  
                  {/* Right: Status indicator */}
                  <FlexItem>
                    {isChunked && isConfirmed && (
                      <CheckCircleIcon style={{ color: '#3e8635' }} />
                    )}
                  </FlexItem>
                </div>
                
                {/* Inline Chunking Configuration */}
                {showInlineConfig && (
                  <div style={{
                    padding: '1rem',
                    backgroundColor: isChunked && !isConfirmed ? '#fff8e6' : '#e7f1fa',
                    border: isChunked && !isConfirmed ? '2px solid #f0ab00' : '2px solid #0066cc',
                    borderTop: 'none',
                    borderRadius: isPreviewExpanded ? '0' : '0 0 6px 6px',
                  }}>
                    <div style={{ fontWeight: '600', marginBottom: '0.75rem', color: isChunked && !isConfirmed ? '#856404' : '#0066cc' }}>
                      {isChunked && !isConfirmed ? 'Adjust Chunking Settings' : 'Configure Chunking'} for {fileName}
                      {isChunked && !isConfirmed && (
                        <span style={{ fontWeight: 'normal', fontSize: '0.85rem', marginLeft: '0.5rem' }}>
                          {needsReApply 
                            ? '- Click Re-Apply to preview changes, then Confirm' 
                            : '- Review the preview below and click Confirm when satisfied'}
                        </span>
                      )}
                    </div>
                    <Grid hasGutter>
                      <GridItem span={5}>
                        <div>
                          <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Chunk Size (words)</div>
                          <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                            Number of words per chunk. Recommended: 500-1500.
                          </div>
                          <NumberInput
                            id={`chunk-size-${index}`}
                            value={chunkSize}
                            onMinus={() => setChunkSize(Math.max(100, chunkSize - 100))}
                            onPlus={() => setChunkSize(chunkSize + 100)}
                            onChange={(e) => setChunkSize(parseInt(e.target.value) || 1000)}
                            min={100}
                            max={5000}
                          />
                        </div>
                      </GridItem>
                      <GridItem span={5}>
                        <div>
                          <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Chunk Overlap (words)</div>
                          {chunkOverlap >= chunkSize ? (
                            <div style={{ fontSize: '0.75rem', color: '#c9190b', marginBottom: '0.5rem' }}>
                              Overlap must be less than chunk size.
                            </div>
                          ) : (
                            <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                              Must be less than chunk size. Recommended: 10-20% of chunk size.
                            </div>
                          )}
                          <NumberInput
                            id={`chunk-overlap-${index}`}
                            value={chunkOverlap}
                            onMinus={() => setChunkOverlap(Math.max(0, chunkOverlap - 50))}
                            onPlus={() => setChunkOverlap(chunkOverlap + 50)}
                            onChange={(e) => setChunkOverlap(parseInt(e.target.value) || 100)}
                            min={0}
                            max={chunkSize - 1}
                          />
                        </div>
                      </GridItem>
                      <GridItem span={2} />
                      <GridItem span={12}>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <Button 
                            variant="secondary" 
                            onClick={() => handleChunk([fileName])}
                            isLoading={isChunking}
                            isDisabled={isChunking || chunkOverlap >= chunkSize}
                          >
                            {isChunking ? 'Applying...' : (isChunked ? 'Re-Apply' : 'Apply')}
                          </Button>
                          {isChunked && !isConfirmed && (
                            <Tooltip 
                              content={needsReApply ? "Click Re-Apply first to confirm your changes" : "Confirm chunking settings"}
                              trigger={needsReApply ? "mouseenter" : "manual"}
                            >
                              <Button 
                                variant="primary" 
                                onClick={() => {
                                  setFileChunkConfigs(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], isConfirmed: true, needsReApply: false }
                                  }));
                                  setExpandedChunkPreviews(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(fileName);
                                    return newSet;
                                  });
                                  setSelectedFilesForChunking(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(fileName);
                                    return newSet;
                                  });
                                }}
                                isDisabled={needsReApply}
                              >
                                Confirm
                              </Button>
                            </Tooltip>
                          )}
                        </div>
                      </GridItem>
                    </Grid>
                  </div>
                )}
                
                {/* Inline Preview */}
                {isPreviewExpanded && isChunked && (
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#f5f5f5',
                    border: !isConfirmed ? '2px solid #f0ab00' : '1px solid #3e8635',
                    borderTop: 'none',
                    borderRadius: '0 0 6px 6px',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontWeight: '600', color: !isConfirmed ? '#f0ab00' : '#3e8635' }}>
                        <CubesIcon style={{ marginRight: '0.5rem' }} />
                        Chunk Preview ({fileConfig.totalChunks} chunks)
                        {!isConfirmed && <Badge style={{ marginLeft: '0.5rem', backgroundColor: '#fff3cd', color: '#856404' }}>
                          {needsReApply ? 'Click Re-Apply to update' : 'Pending Confirmation'}
                        </Badge>}
                      </span>
                    </div>
                    <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                      {(fileConfig.chunks || []).slice(0, 3).map((chunk, idx) => (
                        <div 
                          key={idx} 
                          style={{ 
                            marginBottom: '0.75rem',
                            padding: '0.75rem',
                            backgroundColor: 'white',
                            borderRadius: '4px',
                            border: '1px solid #d2d2d2',
                            cursor: 'pointer',
                            transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
                          }}
                          onClick={() => {
                            setSelectedChunkForModal({ chunk, index: idx, fileName });
                            setChunkModalOpen(true);
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = '#0066cc';
                            e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,102,204,0.2)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = '#d2d2d2';
                            e.currentTarget.style.boxShadow = 'none';
                          }}
                        >
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            marginBottom: '0.5rem',
                            fontSize: '0.85rem',
                            color: '#6a6e73'
                          }}>
                            <span>Chunk {(chunk.chunk_index || idx) + 1}</span>
                            <span>{chunk.word_count || chunk.document?.split(/\s+/).length || 0} words · Click to view full</span>
                          </div>
                          <div style={{ 
                            fontFamily: 'monospace', 
                            fontSize: '0.8rem',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '100px',
                            overflowY: 'hidden',
                            color: '#333',
                          }}>
                            {chunk.document?.substring(0, 300)}
                            {chunk.document?.length > 300 && '...'}
                          </div>
                        </div>
                      ))}
                      {(fileConfig.chunks || []).length > 3 && (
                        <div style={{ fontSize: '0.85rem', color: '#6a6e73', fontStyle: 'italic' }}>
                          Showing first 3 of {fileConfig.totalChunks} chunks · Click any chunk to view full content
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {/* Multi-file Chunking Configuration */}
        {selectedFilesForChunking.size > 1 && (
          <Card isFlat style={{ marginBottom: '1rem', backgroundColor: '#e7f1fa', border: '1px solid #0066cc' }}>
            <CardBody>
              <Title headingLevel="h4" size="md" style={{ marginBottom: '1rem' }}>
                {selectedFilesForChunking.size === convertedFiles.length 
                  ? 'Configure Chunking for All Files'
                  : `Configure Chunking for ${selectedFilesForChunking.size} Selected Files`}
              </Title>
              <Grid hasGutter>
                <GridItem span={5}>
                  <div>
                    <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Chunk Size (words)</div>
                    <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                      Number of words per chunk. Recommended: 500-1500.
                    </div>
                    <NumberInput
                      id="chunk-size"
                      value={chunkSize}
                      onMinus={() => setChunkSize(Math.max(100, chunkSize - 100))}
                      onPlus={() => setChunkSize(chunkSize + 100)}
                      onChange={(e) => setChunkSize(parseInt(e.target.value) || 1000)}
                      min={100}
                      max={5000}
                    />
                  </div>
                </GridItem>
                <GridItem span={5}>
                  <div>
                    <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Chunk Overlap (words)</div>
                    {chunkOverlap >= chunkSize ? (
                      <div style={{ fontSize: '0.75rem', color: '#c9190b', marginBottom: '0.5rem' }}>
                        Overlap must be less than chunk size.
                      </div>
                    ) : (
                      <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                        Must be less than chunk size. Recommended: 10-20% of chunk size.
                      </div>
                    )}
                    <NumberInput
                      id="chunk-overlap"
                      value={chunkOverlap}
                      onMinus={() => setChunkOverlap(Math.max(0, chunkOverlap - 50))}
                      onPlus={() => setChunkOverlap(chunkOverlap + 50)}
                      onChange={(e) => setChunkOverlap(parseInt(e.target.value) || 100)}
                      min={0}
                      max={chunkSize - 1}
                    />
                  </div>
                </GridItem>
                <GridItem span={2} style={{ display: 'flex', alignItems: 'flex-end' }}>
                  <Button 
                    variant="primary" 
                    onClick={() => handleChunk()}
                    isLoading={isChunking}
                    isDisabled={isChunking || chunkOverlap >= chunkSize}
                    style={{ width: '100%' }}
                  >
                    {isChunking ? 'Chunking...' : 'Apply'}
                  </Button>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        )}
      </CardBody>}
    </Card>
  );
};

export default ChunkingSettings;
