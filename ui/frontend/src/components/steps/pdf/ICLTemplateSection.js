import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Button,
  Alert,
  AlertVariant,
  Badge,
  Grid,
  GridItem,
  TextInput,
  TextArea,
  FormGroup,
  ExpandableSection,
  Popover,
  NumberInput,
  Spinner,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  FileIcon,
  OutlinedQuestionCircleIcon,
} from '@patternfly/react-icons';
import { Flex, FlexItem } from '@patternfly/react-core';

/**
 * Step 4: ICL Template (per-file)
 */
const ICLTemplateSection = ({
  expandedSteps,
  toggleStep,
  filesReadyForNextStep,
  needsICL,
  fileICLConfigs,
  editingFileICL,
  expandedICLFiles,
  fileBasicInfo,
  fileChunkConfigs,
  iclTemplates,
  filesWithoutBasicInfoCount,
  selectedChunkIdx,
  loadingChunks,
  setEditingFileICL,
  setFileICLConfigs,
  setExpandedICLFiles,
  setUseCustomICL,
  setCustomICL,
  setSelectedTemplateIndex,
  loadChunkForFile,
  SDGTermTooltip,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step4')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {Object.values(fileICLConfigs).some(c => c.isComplete) ? (
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
              }}>4</span>
            )}
            <SDGTermTooltip term="icl_template">
              ICL Template
            </SDGTermTooltip>
            <Badge isRead style={{ marginLeft: '0.5rem' }}>Required</Badge>
            {!expandedSteps.step4 && Object.values(fileICLConfigs).some(c => c.isComplete) && (
              <Badge isRead style={{ marginLeft: '0.5rem' }}>
                {Object.values(fileICLConfigs).filter(c => c.isComplete).length} file{Object.values(fileICLConfigs).filter(c => c.isComplete).length !== 1 ? 's' : ''} configured
              </Badge>
            )}
          </div>
          <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            {expandedSteps.step4 ? '▼' : '▶'}
          </span>
        </div>
      </CardTitle>
      {expandedSteps.step4 && <CardBody>
        {/* Warning about files without basic info */}
        {filesWithoutBasicInfoCount > 0 && (
          <Alert variant={AlertVariant.warning} isInline title={`${filesWithoutBasicInfoCount} file(s) without basic info`} style={{ marginBottom: '1rem' }}>
            Some chunked files don't have domain and document outline configured in Step 3.
          </Alert>
        )}
        
        <Alert variant={AlertVariant.info} isInline title="In-Context Learning (ICL) Examples Required" style={{ marginBottom: '1rem' }}>
          Configure ICL examples for each file. You can use a pre-built template or create your own custom examples per file.
        </Alert>
        
        {/* Per-file ICL configuration */}
        {Object.entries(fileBasicInfo)
          .filter(([_, info]) => info.isComplete)
          .map(([fileName, _], fileIndex) => {
            const savedICL = fileICLConfigs[fileName] || { useCustom: false, templateIndex: null, customICL: {}, isComplete: false };
            const editingICL = editingFileICL[fileName] || {};
            const isComplete = savedICL.isComplete;
            const currentUseCustom = editingICL.useCustom ?? savedICL.useCustom;
            const currentTemplateIndex = editingICL.templateIndex ?? savedICL.templateIndex;
            const currentCustomICL = {
              icl_document: editingICL.icl_document ?? savedICL.customICL?.icl_document ?? '',
              icl_query_1: editingICL.icl_query_1 ?? savedICL.customICL?.icl_query_1 ?? '',
              icl_response_1: editingICL.icl_response_1 ?? savedICL.customICL?.icl_response_1 ?? '',
              icl_query_2: editingICL.icl_query_2 ?? savedICL.customICL?.icl_query_2 ?? '',
              icl_response_2: editingICL.icl_response_2 ?? savedICL.customICL?.icl_response_2 ?? '',
              icl_query_3: editingICL.icl_query_3 ?? savedICL.customICL?.icl_query_3 ?? '',
              icl_response_3: editingICL.icl_response_3 ?? savedICL.customICL?.icl_response_3 ?? '',
            };
            const hasChanges = Object.keys(editingICL).length > 0;
            const canSave = currentUseCustom 
              ? (currentCustomICL.icl_document && currentCustomICL.icl_query_1 && currentCustomICL.icl_query_2 && currentCustomICL.icl_query_3)
              : currentTemplateIndex !== null;
            const isExpanded = expandedICLFiles.has(fileName) || !isComplete || hasChanges;
            
            return (
              <div 
                key={fileName}
                style={{
                  marginBottom: '0.5rem',
                  backgroundColor: isComplete && !isExpanded ? '#f0fff0' : (hasChanges ? '#fffbf0' : '#fafafa'),
                  borderRadius: '6px',
                  border: isComplete ? '1px solid #3e8635' : (hasChanges ? '1px solid #f0ab00' : '1px solid #e0e0e0'),
                  overflow: 'hidden',
                }}
              >
                {/* File Header */}
                <div 
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center', 
                    padding: '0.75rem 1rem',
                    cursor: isComplete && !hasChanges ? 'pointer' : 'default',
                    backgroundColor: isComplete && !isExpanded ? '#f0fff0' : 'transparent',
                  }}
                  onClick={() => {
                    if (isComplete && !hasChanges) {
                      setExpandedICLFiles(prev => {
                        const newSet = new Set(prev);
                        if (newSet.has(fileName)) {
                          newSet.delete(fileName);
                        } else {
                          newSet.add(fileName);
                        }
                        return newSet;
                      });
                    }
                  }}
                >
                  <Flex alignItems={{ default: 'alignItemsCenter' }}>
                    <FlexItem>
                      <FileIcon style={{ marginRight: '0.5rem', color: isComplete ? '#3e8635' : '#0066cc' }} />
                    </FlexItem>
                    <FlexItem>
                      <span style={{ fontWeight: '600' }}>{fileName}</span>
                    </FlexItem>
                  </Flex>
                  <Flex style={{ gap: '0.5rem', alignItems: 'center' }}>
                    {hasChanges && (
                      <Badge style={{ backgroundColor: '#fffbf0', color: '#f0ab00', border: '1px solid #f0ab00' }}>
                        Unsaved changes
                      </Badge>
                    )}
                    {isComplete && !hasChanges && (
                      <Badge style={{ backgroundColor: '#f0fff0', color: '#3e8635', border: '1px solid #3e8635' }}>
                        <CheckCircleIcon style={{ marginRight: '0.25rem' }} />
                        ICL Configured
                      </Badge>
                    )}
                    {isComplete && !hasChanges && (
                      <span style={{ fontSize: '0.875rem', color: '#6a6e73', marginLeft: '0.5rem' }}>
                        {isExpanded ? '▼' : '▶'}
                      </span>
                    )}
                  </Flex>
                </div>
                
                {/* Expandable Content */}
                {isExpanded && (
                  <div style={{ padding: '0 1rem 1rem 1rem' }}>
                    {/* Config summary when complete */}
                    {isComplete && !hasChanges && (
                      <div style={{ 
                        marginBottom: '1rem', 
                        padding: '0.5rem 0.75rem', 
                        backgroundColor: '#e8f5e9', 
                        borderRadius: '4px',
                        fontSize: '0.9rem'
                      }}>
                        <strong>Current config:</strong> {savedICL.useCustom ? 'Custom ICL' : `Template: ${iclTemplates[savedICL.templateIndex]?.name || 'Selected'}`}
                      </div>
                    )}
                    
                    {/* Template vs Custom Toggle */}
                    <div style={{ marginBottom: '1rem' }}>
                      <Button
                        variant={!currentUseCustom ? 'primary' : 'secondary'}
                        isSmall
                        onClick={() => {
                          setEditingFileICL(prev => ({
                            ...prev,
                            [fileName]: { ...prev[fileName], useCustom: false }
                          }));
                        }}
                        style={{ marginRight: '0.5rem' }}
                      >
                        Use Template
                      </Button>
                      <Button
                        variant={currentUseCustom ? 'primary' : 'secondary'}
                        isSmall
                        onClick={() => {
                          setEditingFileICL(prev => ({
                            ...prev,
                            [fileName]: { ...prev[fileName], useCustom: true }
                          }));
                        }}
                      >
                        Create Custom
                      </Button>
                    </div>
                    
                    {/* Template Selection or Custom Form */}
                    {!currentUseCustom ? (
                      <div style={{ marginBottom: '1rem' }}>
                        <p style={{ fontWeight: '500', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Select Template:</p>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                          {iclTemplates.map((template, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: '0.75rem',
                                border: currentTemplateIndex === idx ? '2px solid #0066cc' : '1px solid #d2d2d2',
                                borderRadius: '4px',
                                backgroundColor: currentTemplateIndex === idx ? '#e7f1fa' : '#fff',
                              }}
                            >
                              <div 
                                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}
                                onClick={() => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: false, templateIndex: idx }
                                  }));
                                }}
                              >
                                <div style={{
                                  width: '16px',
                                  height: '16px',
                                  borderRadius: '50%',
                                  border: currentTemplateIndex === idx ? '5px solid #0066cc' : '2px solid #6a6e73',
                                  backgroundColor: '#fff',
                                  flexShrink: 0,
                                }} />
                                <div style={{ flex: 1 }}>
                                  <span style={{ fontWeight: '500' }}>{template.name}</span>
                                  <span style={{ color: '#6a6e73', fontSize: '0.85rem', marginLeft: '0.5rem' }}>
                                    {template.description}
                                  </span>
                                </div>
                              </div>
                              {/* Expandable template details */}
                              <ExpandableSection 
                                toggleText={`View ${template.name} details`}
                                style={{ marginTop: '0.5rem' }}
                              >
                                <div style={{ 
                                  backgroundColor: '#f5f5f5', 
                                  padding: '0.75rem', 
                                  borderRadius: '4px',
                                  fontSize: '0.85rem',
                                  marginTop: '0.5rem'
                                }}>
                                  <div style={{ marginBottom: '0.75rem' }}>
                                    <strong>ICL Document:</strong>
                                    <div style={{ 
                                      whiteSpace: 'pre-wrap', 
                                      fontFamily: 'monospace', 
                                      fontSize: '0.8rem',
                                      backgroundColor: '#fff',
                                      padding: '0.5rem',
                                      borderRadius: '4px',
                                      marginTop: '0.25rem',
                                      maxHeight: '150px',
                                      overflowY: 'auto'
                                    }}>
                                      {template.template?.icl_document?.substring(0, 500)}
                                      {template.template?.icl_document?.length > 500 && '...'}
                                    </div>
                                  </div>
                                  <Grid hasGutter>
                                    <GridItem span={6}>
                                      <strong>Q1:</strong> {template.template?.icl_query_1}
                                    </GridItem>
                                    <GridItem span={6}>
                                      <strong>A1:</strong> {template.template?.icl_response_1?.substring(0, 100)}{template.template?.icl_response_1?.length > 100 && '...'}
                                    </GridItem>
                                    <GridItem span={6}>
                                      <strong>Q2:</strong> {template.template?.icl_query_2}
                                    </GridItem>
                                    <GridItem span={6}>
                                      <strong>A2:</strong> {template.template?.icl_response_2?.substring(0, 100)}{template.template?.icl_response_2?.length > 100 && '...'}
                                    </GridItem>
                                    <GridItem span={6}>
                                      <strong>Q3:</strong> {template.template?.icl_query_3}
                                    </GridItem>
                                    <GridItem span={6}>
                                      <strong>A3:</strong> {template.template?.icl_response_3?.substring(0, 100)}{template.template?.icl_response_3?.length > 100 && '...'}
                                    </GridItem>
                                  </Grid>
                                </div>
                              </ExpandableSection>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div style={{ marginBottom: '1rem' }}>
                        {/* Example Reference Popover */}
                        {iclTemplates.length > 0 && (
                          <div style={{ marginBottom: '0.75rem' }}>
                            <Popover
                              headerContent="Example ICL Template"
                              bodyContent={
                                <div style={{ maxWidth: '400px', fontSize: '0.85rem' }}>
                                  <div style={{ marginBottom: '0.75rem' }}>
                                    <strong>ICL Document:</strong>
                                    <div style={{ 
                                      backgroundColor: '#f5f5f5', 
                                      padding: '0.5rem', 
                                      borderRadius: '4px',
                                      marginTop: '0.25rem',
                                      maxHeight: '100px',
                                      overflowY: 'auto',
                                      fontSize: '0.8rem'
                                    }}>
                                      {iclTemplates[0]?.template?.icl_document?.substring(0, 300)}
                                      {iclTemplates[0]?.template?.icl_document?.length > 300 && '...'}
                                    </div>
                                  </div>
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                                    <div><strong>Q1:</strong> {iclTemplates[0]?.template?.icl_query_1?.substring(0, 50)}{iclTemplates[0]?.template?.icl_query_1?.length > 50 && '...'}</div>
                                    <div><strong>A1:</strong> {iclTemplates[0]?.template?.icl_response_1?.substring(0, 50)}{iclTemplates[0]?.template?.icl_response_1?.length > 50 && '...'}</div>
                                    <div><strong>Q2:</strong> {iclTemplates[0]?.template?.icl_query_2?.substring(0, 50)}{iclTemplates[0]?.template?.icl_query_2?.length > 50 && '...'}</div>
                                    <div><strong>A2:</strong> {iclTemplates[0]?.template?.icl_response_2?.substring(0, 50)}{iclTemplates[0]?.template?.icl_response_2?.length > 50 && '...'}</div>
                                    <div><strong>Q3:</strong> {iclTemplates[0]?.template?.icl_query_3?.substring(0, 50)}{iclTemplates[0]?.template?.icl_query_3?.length > 50 && '...'}</div>
                                    <div><strong>A3:</strong> {iclTemplates[0]?.template?.icl_response_3?.substring(0, 50)}{iclTemplates[0]?.template?.icl_response_3?.length > 50 && '...'}</div>
                                  </div>
                                </div>
                              }
                              position="right"
                            >
                              <Button variant="link" isSmall style={{ padding: 0 }}>
                                <OutlinedQuestionCircleIcon style={{ marginRight: '0.25rem' }} />
                                See Example for Reference
                              </Button>
                            </Popover>
                          </div>
                        )}
                        
                        <FormGroup 
                          label={
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                              <span>ICL Document *</span>
                              {(fileChunkConfigs[fileName]?.chunks?.length > 0 || fileChunkConfigs[fileName]?.totalChunks > 0) && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                  <Button
                                    variant="link"
                                    isSmall
                                    isInline
                                    style={{ fontSize: '0.8rem', padding: 0, whiteSpace: 'nowrap' }}
                                    onClick={() => {
                                      loadChunkForFile(fileName, selectedChunkIdx[fileName] ?? 0);
                                    }}
                                  >
                                    Load chunk from dataset:
                                  </Button>
                                  <NumberInput
                                    value={(selectedChunkIdx[fileName] ?? 0) + 1}
                                    min={1}
                                    max={fileChunkConfigs[fileName]?.totalChunks || fileChunkConfigs[fileName]?.chunks?.length || 1}
                                    onMinus={() => {
                                      const currentIdx = selectedChunkIdx[fileName] ?? 0;
                                      if (currentIdx > 0) {
                                        loadChunkForFile(fileName, currentIdx - 1);
                                      }
                                    }}
                                    onPlus={() => {
                                      const currentIdx = selectedChunkIdx[fileName] ?? 0;
                                      const maxIdx = (fileChunkConfigs[fileName]?.totalChunks || fileChunkConfigs[fileName]?.chunks?.length || 1) - 1;
                                      if (currentIdx < maxIdx) {
                                        loadChunkForFile(fileName, currentIdx + 1);
                                      }
                                    }}
                                    onChange={(event) => {
                                      const val = Number(event.target.value);
                                      const maxChunks = fileChunkConfigs[fileName]?.totalChunks || fileChunkConfigs[fileName]?.chunks?.length || 1;
                                      if (!isNaN(val) && val >= 1 && val <= maxChunks) {
                                        loadChunkForFile(fileName, val - 1);
                                      }
                                    }}
                                    inputName={`chunk-idx-${fileIndex}`}
                                    inputAriaLabel="Chunk index"
                                    minusBtnAriaLabel="Minus"
                                    plusBtnAriaLabel="Plus"
                                    widthChars={3}
                                    style={{ '--pf-c-number-input__input--c-form-control--Width': '60px' }}
                                  />
                                  <span style={{ fontSize: '0.75rem', color: '#6a6e73', whiteSpace: 'nowrap' }}>
                                    of {fileChunkConfigs[fileName]?.totalChunks || fileChunkConfigs[fileName]?.chunks?.length || 0}
                                  </span>
                                  {loadingChunks[fileName] && (
                                    <Spinner size="sm" aria-label="Loading chunk" />
                                  )}
                                </div>
                              )}
                            </div>
                          }
                          fieldId={`icl-doc-${fileIndex}`} 
                          style={{ marginBottom: '0.75rem' }}
                        >
                          {/* Warning if document is too long */}
                          {currentCustomICL.icl_document && currentCustomICL.icl_document.split(/\s+/).length > 250 && (
                            <Alert 
                              variant={AlertVariant.warning} 
                              isInline 
                              title="Document may be too long"
                              style={{ marginBottom: '0.5rem' }}
                            >
                              This document has {currentCustomICL.icl_document.split(/\s+/).length} words. 
                              Consider reducing to under 250 words to avoid overwhelming the prompts.
                            </Alert>
                          )}
                          <TextArea
                            id={`icl-doc-${fileIndex}`}
                            value={currentCustomICL.icl_document}
                            onChange={(_, value) => {
                              setEditingFileICL(prev => ({
                                ...prev,
                                [fileName]: { ...prev[fileName], useCustom: true, icl_document: value }
                              }));
                            }}
                            placeholder="Enter an example document excerpt..."
                            rows={3}
                          />
                          {/* Word count indicator */}
                          {currentCustomICL.icl_document && (
                            <div style={{ 
                              fontSize: '0.75rem', 
                              color: currentCustomICL.icl_document.split(/\s+/).length > 250 ? '#c9190b' : '#6a6e73',
                              marginTop: '0.25rem'
                            }}>
                              {currentCustomICL.icl_document.split(/\s+/).length} words • Recommended: under 250 words
                            </div>
                          )}
                        </FormGroup>
                        
                        <Grid hasGutter style={{ marginBottom: '0.5rem' }}>
                          <GridItem span={6}>
                            <FormGroup label="Q1" fieldId={`q1-${fileIndex}`} isRequired>
                              <TextInput
                                id={`q1-${fileIndex}`}
                                value={currentCustomICL.icl_query_1}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_query_1: value }
                                  }));
                                }}
                                placeholder="Question 1..."
                              />
                            </FormGroup>
                          </GridItem>
                          <GridItem span={6}>
                            <FormGroup label="A1" fieldId={`a1-${fileIndex}`}>
                              <TextInput
                                id={`a1-${fileIndex}`}
                                value={currentCustomICL.icl_response_1}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_response_1: value }
                                  }));
                                }}
                                placeholder="Answer 1..."
                              />
                            </FormGroup>
                          </GridItem>
                        </Grid>
                        
                        <Grid hasGutter style={{ marginBottom: '0.5rem' }}>
                          <GridItem span={6}>
                            <FormGroup label="Q2" fieldId={`q2-${fileIndex}`} isRequired>
                              <TextInput
                                id={`q2-${fileIndex}`}
                                value={currentCustomICL.icl_query_2}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_query_2: value }
                                  }));
                                }}
                                placeholder="Question 2..."
                              />
                            </FormGroup>
                          </GridItem>
                          <GridItem span={6}>
                            <FormGroup label="A2" fieldId={`a2-${fileIndex}`}>
                              <TextInput
                                id={`a2-${fileIndex}`}
                                value={currentCustomICL.icl_response_2}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_response_2: value }
                                  }));
                                }}
                                placeholder="Answer 2..."
                              />
                            </FormGroup>
                          </GridItem>
                        </Grid>
                        
                        <Grid hasGutter>
                          <GridItem span={6}>
                            <FormGroup label="Q3" fieldId={`q3-${fileIndex}`} isRequired>
                              <TextInput
                                id={`q3-${fileIndex}`}
                                value={currentCustomICL.icl_query_3}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_query_3: value }
                                  }));
                                }}
                                placeholder="Question 3..."
                              />
                            </FormGroup>
                          </GridItem>
                          <GridItem span={6}>
                            <FormGroup label="A3" fieldId={`a3-${fileIndex}`}>
                              <TextInput
                                id={`a3-${fileIndex}`}
                                value={currentCustomICL.icl_response_3}
                                onChange={(_, value) => {
                                  setEditingFileICL(prev => ({
                                    ...prev,
                                    [fileName]: { ...prev[fileName], useCustom: true, icl_response_3: value }
                                  }));
                                }}
                                placeholder="Answer 3..."
                              />
                            </FormGroup>
                          </GridItem>
                        </Grid>
                      </div>
                    )}
                    
                    {/* Save Button */}
                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                      <Button
                        variant="primary"
                        isSmall
                        onClick={() => {
                          if (canSave) {
                            setFileICLConfigs(prev => ({
                              ...prev,
                              [fileName]: {
                                useCustom: currentUseCustom,
                                templateIndex: currentUseCustom ? null : currentTemplateIndex,
                                customICL: currentUseCustom ? currentCustomICL : {},
                                isComplete: true,
                              }
                            }));
                            setEditingFileICL(prev => {
                              const newState = { ...prev };
                              delete newState[fileName];
                              return newState;
                            });
                            setExpandedICLFiles(prev => {
                              const newSet = new Set(prev);
                              newSet.delete(fileName);
                              return newSet;
                            });
                            if (currentUseCustom) {
                              setUseCustomICL(true);
                              setCustomICL(currentCustomICL);
                            } else {
                              setUseCustomICL(false);
                              setSelectedTemplateIndex(currentTemplateIndex);
                            }
                          }
                        }}
                        isDisabled={!canSave}
                      >
                        Save ICL Config
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        
        {/* Summary */}
        {Object.values(fileICLConfigs).some(c => c.isComplete) && (
          <Alert 
            variant={AlertVariant.success} 
            isInline 
            title={`${Object.values(fileICLConfigs).filter(c => c.isComplete).length} of ${Object.values(fileBasicInfo).filter(info => info.isComplete).length} files have ICL configured`}
            style={{ marginTop: '0.5rem' }}
          >
            Files with ICL configuration are ready for dataset creation.
          </Alert>
        )}
      </CardBody>}
    </Card>
  );
};

export default ICLTemplateSection;
