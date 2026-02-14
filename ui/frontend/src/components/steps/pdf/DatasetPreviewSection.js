import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Button,
  Alert,
  Badge,
  Grid,
  GridItem,
  NumberInput,
  TextInput,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  FileIcon,
  TimesIcon,
} from '@patternfly/react-icons';

/**
 * Step 5: Dataset Preview
 */
const DatasetPreviewSection = ({
  expandedSteps,
  toggleStep,
  needsICL,
  needsBasicInfoStep,
  needsDomain,
  needsDocumentOutline,
  contentColumnName,
  fileICLConfigs,
  fileBasicInfo,
  fileChunkConfigs,
  iclTemplates,
  filesReadyForNextStep,
  previewSelectedColumns,
  previewSelectedFiles,
  previewSamplesPerFile,
  previewFileSearchQuery,
  previewFileSearchFocused,
  previewExpandedFiles,
  previewAutoExpanded,
  setPreviewSelectedColumns,
  setPreviewSelectedFiles,
  setPreviewSamplesPerFile,
  setPreviewFileSearchQuery,
  setPreviewFileSearchFocused,
  setPreviewExpandedFiles,
  setPreviewAutoExpanded,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step5')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircleIcon color="var(--pf-global--success-color--100)" />
            Dataset Preview
            <Badge isRead style={{ marginLeft: '0.5rem' }}>
              {needsICL 
                ? `${Object.values(fileICLConfigs).filter(c => c.isComplete).length} files configured`
                : (needsBasicInfoStep 
                    ? `${Object.values(fileBasicInfo).filter(info => info.isComplete).length} files configured`
                    : `${Object.values(fileChunkConfigs).filter(c => c.isChunked).length} files ready`)
              }
            </Badge>
          </div>
          <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            {expandedSteps.step5 ? '▼' : '▶'}
          </span>
        </div>
      </CardTitle>
      {expandedSteps.step5 && <CardBody>
        <p style={{ marginBottom: '1rem', color: '#6a6e73' }}>
          Preview how your dataset will look. Each chunk from your documents becomes a row with the configured metadata.
        </p>
        
        {/* Generate preview samples from configured files */}
        {(() => {
          // Build column names dynamically based on flow requirements
          const columnNames = [contentColumnName];
          if (needsDomain) columnNames.push('domain');
          if (needsDocumentOutline) columnNames.push('document_outline');
          if (needsICL) {
            columnNames.push('icl_document', 'icl_query_1', 'icl_response_1', 'icl_query_2', 'icl_response_2', 'icl_query_3', 'icl_response_3');
          }
          
          const configuredFiles = needsICL 
            ? Object.entries(fileICLConfigs).filter(([_, config]) => config.isComplete)
            : (needsBasicInfoStep 
                ? Object.entries(fileBasicInfo).filter(([_, info]) => info.isComplete)
                : Object.entries(fileChunkConfigs).filter(([_, config]) => config.isChunked).map(([fn, _]) => [fn, { isComplete: true }]));
          
          const activePreviewFiles = previewSelectedFiles.size > 0 
            ? configuredFiles.filter(([fileName]) => previewSelectedFiles.has(fileName))
            : configuredFiles;
          
          const previewSamples = [];
          activePreviewFiles.forEach(([fileName, config]) => {
              const chunkConfig = fileChunkConfigs[fileName];
              const basicInfo = fileBasicInfo[fileName];
              const iclConfig = fileICLConfigs[fileName];
              
              const isFileReady = needsBasicInfoStep ? basicInfo?.isComplete : chunkConfig?.isChunked;
              
              if (chunkConfig?.chunks && isFileReady) {
                const fileSamples = (chunkConfig.chunks || []).slice(0, previewSamplesPerFile).map((chunk, idx) => {
                  const sample = {
                    _source: fileName,
                    _chunkIdx: idx,
                    [contentColumnName]: chunk.document || `[Chunk ${idx + 1}]`,
                  };
                  
                  if (needsDomain) {
                    sample.domain = basicInfo?.domain || '';
                  }
                  
                  if (needsDocumentOutline) {
                    sample.document_outline = basicInfo?.documentOutline || '';
                  }
                  
                  if (needsICL && iclConfig?.isComplete) {
                    const iclData = iclConfig.useCustom 
                      ? iclConfig.customICL 
                      : (iclTemplates[iclConfig.templateIndex]?.template || {});
                    sample.icl_document = iclData.icl_document || '';
                    sample.icl_query_1 = iclData.icl_query_1 || '';
                    sample.icl_response_1 = iclData.icl_response_1 || '';
                    sample.icl_query_2 = iclData.icl_query_2 || '';
                    sample.icl_response_2 = iclData.icl_response_2 || '';
                    sample.icl_query_3 = iclData.icl_query_3 || '';
                    sample.icl_response_3 = iclData.icl_response_3 || '';
                  }
                  
                  return sample;
                });
                previewSamples.push(...fileSamples);
              }
            });
          
          const totalChunks = configuredFiles.reduce((sum, [fileName]) => {
              return sum + (fileChunkConfigs[fileName]?.totalChunks || 0);
            }, 0);
          
          if (configuredFiles.length === 0) {
            const requiredSteps = ['chunking'];
            if (needsBasicInfoStep) requiredSteps.push('basic info');
            if (needsICL) requiredSteps.push('ICL configuration');
            return (
              <Alert variant="info" isInline title="No preview available">
                Complete {requiredSteps.join(' and ')} for at least one file to see a preview.
              </Alert>
            );
          }
          
          const filteredFiles = configuredFiles.filter(([fileName]) => 
            fileName.toLowerCase().includes(previewFileSearchQuery.toLowerCase())
          );
          
          if (!previewAutoExpanded && activePreviewFiles.length > 0) {
            setPreviewAutoExpanded(true);
          }
          
          return (
            <div>
              {/* File Selection Controls */}
              <div style={{ 
                marginBottom: '1.5rem', 
                padding: '1.25rem', 
                backgroundColor: '#fafafa', 
                borderRadius: '12px',
                border: '1px solid #e0e0e0'
              }}>
                <Grid hasGutter>
                  <GridItem span={9}>
                    <div style={{ fontWeight: 600, marginBottom: '0.25rem', fontSize: '0.95rem', color: '#333' }}>
                      Select Files to Preview
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#6a6e73', marginBottom: '0.75rem' }}>
                      This preview is for convenience only. All configured files will be combined into a single dataset.
                    </div>
                    
                    {/* Search Input with Autocomplete */}
                    <div style={{ position: 'relative', marginBottom: '0.75rem' }}>
                      <TextInput
                        type="text"
                        placeholder="Search and add files..."
                        value={previewFileSearchQuery}
                        onChange={(_, value) => setPreviewFileSearchQuery(value)}
                        onFocus={() => setPreviewFileSearchFocused(true)}
                        onBlur={() => setTimeout(() => setPreviewFileSearchFocused(false), 200)}
                        style={{
                          borderRadius: '8px',
                          paddingLeft: '12px',
                        }}
                      />
                      
                      {/* Autocomplete Dropdown */}
                      {previewFileSearchFocused && filteredFiles.length > 0 && (
                        <div style={{
                          position: 'absolute',
                          top: '100%',
                          left: 0,
                          right: 0,
                          backgroundColor: 'white',
                          border: '1px solid #d2d2d2',
                          borderRadius: '8px',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                          zIndex: 1000,
                          maxHeight: '200px',
                          overflowY: 'auto',
                          marginTop: '4px',
                        }}>
                          {filteredFiles.map(([fileName]) => {
                            const isAlreadySelected = previewSelectedFiles.has(fileName);
                            const chunkCount = fileChunkConfigs[fileName]?.totalChunks || 0;
                            return (
                              <div
                                key={fileName}
                                onClick={() => {
                                  if (!isAlreadySelected) {
                                    setPreviewSelectedFiles(prev => new Set([...prev, fileName]));
                                    setPreviewExpandedFiles(prev => new Set([...prev, fileName]));
                                  }
                                  setPreviewFileSearchQuery('');
                                }}
                                style={{
                                  padding: '10px 14px',
                                  cursor: isAlreadySelected ? 'default' : 'pointer',
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  backgroundColor: isAlreadySelected ? '#f0f9ff' : 'transparent',
                                  borderBottom: '1px solid #f0f0f0',
                                  opacity: isAlreadySelected ? 0.7 : 1,
                                }}
                                onMouseEnter={(e) => !isAlreadySelected && (e.currentTarget.style.backgroundColor = '#f5f5f5')}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = isAlreadySelected ? '#f0f9ff' : 'transparent'}
                              >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <FileIcon style={{ color: '#0066cc', fontSize: '0.9rem' }} />
                                  <span style={{ fontSize: '0.9rem' }}>{fileName}</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <Badge isRead style={{ fontSize: '0.75rem' }}>{chunkCount} chunks</Badge>
                                  {isAlreadySelected && (
                                    <CheckCircleIcon style={{ color: '#3e8635', fontSize: '0.9rem' }} />
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                    
                    {/* Selected Files as Tags */}
                    {previewSelectedFiles.size > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
                        {Array.from(previewSelectedFiles).map(fileName => {
                          const chunkCount = fileChunkConfigs[fileName]?.totalChunks || 0;
                          return (
                            <div
                              key={fileName}
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '6px 10px 6px 12px',
                                backgroundColor: '#e7f1fa',
                                border: '1px solid #0066cc',
                                borderRadius: '20px',
                                fontSize: '0.85rem',
                                color: '#0066cc',
                              }}
                            >
                              <FileIcon style={{ fontSize: '0.8rem' }} />
                              <span style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {fileName}
                              </span>
                              <Badge isRead style={{ fontSize: '0.7rem', backgroundColor: '#0066cc', color: 'white' }}>
                                {chunkCount}
                              </Badge>
                              <button
                                onClick={() => {
                                  setPreviewSelectedFiles(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(fileName);
                                    return newSet;
                                  });
                                  setPreviewExpandedFiles(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(fileName);
                                    return newSet;
                                  });
                                }}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  cursor: 'pointer',
                                  padding: '2px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  color: '#0066cc',
                                  borderRadius: '50%',
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#cce4ff'}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                              >
                                <TimesIcon style={{ fontSize: '0.75rem' }} />
                              </button>
                            </div>
                          );
                        })}
                        <Button
                          variant="link"
                          size="sm"
                          onClick={() => {
                            setPreviewSelectedFiles(new Set());
                            setPreviewExpandedFiles(new Set(configuredFiles.map(([fn]) => fn)));
                          }}
                          style={{ fontSize: '0.8rem' }}
                        >
                          Clear All
                        </Button>
                      </div>
                    ) : (
                      <div style={{ 
                        color: '#6a6e73', 
                        fontSize: '0.85rem',
                        fontStyle: 'italic',
                        padding: '8px 0'
                      }}>
                        Showing all {configuredFiles.length} file{configuredFiles.length !== 1 ? 's' : ''}. Use the search to filter specific files.
                      </div>
                    )}
                  </GridItem>
                  
                  <GridItem span={3}>
                    <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem', color: '#333' }}>
                      Samples per File
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <NumberInput
                        value={previewSamplesPerFile}
                        min={1}
                        max={10}
                        onMinus={() => setPreviewSamplesPerFile(prev => Math.max(1, prev - 1))}
                        onPlus={() => setPreviewSamplesPerFile(prev => Math.min(10, prev + 1))}
                        onChange={(e) => {
                          const val = parseInt(e.target.value, 10);
                          if (!isNaN(val) && val >= 1 && val <= 10) {
                            setPreviewSamplesPerFile(val);
                          }
                        }}
                        inputProps={{ style: { textAlign: 'center', width: '50px' } }}
                      />
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginTop: '4px' }}>
                      Max 10 samples per file
                    </div>
                  </GridItem>
                </Grid>
              </div>
              
              {/* Grouped Preview by File */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {activePreviewFiles.map(([fileName], fileIdx) => {
                  const chunkConfig = fileChunkConfigs[fileName];
                  const basicInfo = fileBasicInfo[fileName];
                  const iclConfig = fileICLConfigs[fileName];
                  const isExpanded = previewExpandedFiles.has(fileName);
                  const fileColor = ['#0066cc', '#6753ac', '#009596', '#f0ab00', '#c9190b'][fileIdx % 5];
                  
                  const fileSamples = (chunkConfig?.chunks || []).slice(0, previewSamplesPerFile).map((chunk, idx) => {
                    const sample = {
                      _chunkIdx: idx,
                      [contentColumnName]: chunk.document || `[Chunk ${idx + 1}]`,
                    };
                    if (needsDomain) {
                      sample.domain = basicInfo?.domain || '';
                    }
                    if (needsDocumentOutline) {
                      sample.document_outline = basicInfo?.documentOutline || '';
                    }
                    if (needsICL && iclConfig?.isComplete) {
                      const iclData = iclConfig.useCustom 
                        ? iclConfig.customICL 
                        : (iclTemplates[iclConfig.templateIndex]?.template || {});
                      sample.icl_document = iclData.icl_document || '';
                      sample.icl_query_1 = iclData.icl_query_1 || '';
                      sample.icl_response_1 = iclData.icl_response_1 || '';
                      sample.icl_query_2 = iclData.icl_query_2 || '';
                      sample.icl_response_2 = iclData.icl_response_2 || '';
                      sample.icl_query_3 = iclData.icl_query_3 || '';
                      sample.icl_response_3 = iclData.icl_response_3 || '';
                    }
                    return sample;
                  });
                  
                  return (
                    <div 
                      key={fileName}
                      style={{
                        border: `2px solid ${fileColor}`,
                        borderRadius: '12px',
                        overflow: 'hidden',
                        backgroundColor: 'white',
                      }}
                    >
                      {/* File Header */}
                      <div
                        onClick={() => {
                          setPreviewExpandedFiles(prev => {
                            const newSet = new Set(prev);
                            if (newSet.has(fileName)) {
                              newSet.delete(fileName);
                            } else {
                              newSet.add(fileName);
                            }
                            return newSet;
                          });
                        }}
                        style={{
                          padding: '1rem 1.25rem',
                          backgroundColor: fileColor,
                          color: 'white',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <FileIcon style={{ fontSize: '1.1rem' }} />
                          <span style={{ fontWeight: 600, fontSize: '1rem' }}>{fileName}</span>
                          <Badge style={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white', fontSize: '0.8rem' }}>
                            {chunkConfig?.totalChunks || 0} total chunks
                          </Badge>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <span style={{ fontSize: '0.85rem', opacity: 0.9 }}>
                            {fileSamples.length} sample{fileSamples.length !== 1 ? 's' : ''} shown
                          </span>
                          <span style={{ fontSize: '1.2rem' }}>{isExpanded ? '▼' : '▶'}</span>
                        </div>
                      </div>
                      
                      {/* File Samples */}
                      {isExpanded && (
                        <div style={{ padding: '1rem', backgroundColor: '#fafafa' }}>
                          {fileSamples.map((sample, sampleIdx) => {
                            const globalIdx = `${fileName}_${sampleIdx}`;
                            const selectedColumn = previewSelectedColumns[globalIdx] || contentColumnName;
                            
                            return (
                              <div 
                                key={sampleIdx}
                                style={{
                                  marginBottom: sampleIdx < fileSamples.length - 1 ? '1rem' : 0,
                                  padding: '1rem',
                                  backgroundColor: 'white',
                                  borderRadius: '8px',
                                  border: '1px solid #e0e0e0',
                                }}
                              >
                                {/* Sample Header */}
                                <div style={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  marginBottom: '0.75rem',
                                  paddingBottom: '0.5rem',
                                  borderBottom: `2px solid ${fileColor}20`,
                                }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{
                                      width: '28px',
                                      height: '28px',
                                      borderRadius: '50%',
                                      backgroundColor: fileColor,
                                      color: 'white',
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      fontSize: '0.85rem',
                                      fontWeight: 'bold',
                                    }}>
                                      {sampleIdx + 1}
                                    </span>
                                    <span style={{ fontWeight: 600, color: '#333' }}>
                                      Chunk {sample._chunkIdx + 1}
                                    </span>
                                  </div>
                                </div>
                                
                                {/* Column Selection Tabs */}
                                <div style={{ 
                                  marginBottom: '0.75rem',
                                  display: 'flex',
                                  flexWrap: 'wrap',
                                  gap: '6px'
                                }}>
                                  {columnNames.map(colName => (
                                    <button
                                      key={colName}
                                      onClick={() => setPreviewSelectedColumns(prev => ({
                                        ...prev,
                                        [globalIdx]: colName
                                      }))}
                                      style={{
                                        fontSize: '0.8rem',
                                        padding: '5px 12px',
                                        borderRadius: '6px',
                                        border: 'none',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s',
                                        ...(selectedColumn === colName ? {
                                          backgroundColor: fileColor,
                                          color: 'white',
                                        } : {
                                          backgroundColor: '#f0f0f0',
                                          color: '#555',
                                        })
                                      }}
                                    >
                                      {colName}
                                    </button>
                                  ))}
                                </div>
                                
                                {/* Selected Column Content */}
                                <div style={{
                                  backgroundColor: '#f8f9fa',
                                  borderRadius: '6px',
                                  border: '1px solid #e8e8e8',
                                  overflow: 'hidden'
                                }}>
                                  <div style={{
                                    backgroundColor: `${fileColor}15`,
                                    padding: '8px 12px',
                                    fontWeight: 600,
                                    fontSize: '0.8rem',
                                    color: fileColor,
                                    borderBottom: '1px solid #e8e8e8'
                                  }}>
                                    {selectedColumn}
                                  </div>
                                  <div style={{
                                    padding: '12px',
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    maxHeight: '180px',
                                    overflowY: 'auto',
                                    fontSize: '0.85rem',
                                    lineHeight: '1.6',
                                    color: '#333'
                                  }}>
                                    {sample[selectedColumn] 
                                      ? (typeof sample[selectedColumn] === 'object' 
                                          ? JSON.stringify(sample[selectedColumn], null, 2)
                                          : String(sample[selectedColumn]))
                                      : <span style={{ color: '#999', fontStyle: 'italic' }}>(empty)</span>
                                    }
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </CardBody>}
    </Card>
  );
};

export default DatasetPreviewSection;
