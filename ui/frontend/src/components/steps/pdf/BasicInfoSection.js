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
  Popover,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  FileIcon,
  OutlinedQuestionCircleIcon,
} from '@patternfly/react-icons';
import { Flex, FlexItem } from '@patternfly/react-core';

/**
 * Step 3: Basic Information (per-file domain, document outline)
 */
const BasicInfoSection = ({
  isStep3Complete,
  expandedSteps,
  toggleStep,
  fileChunkConfigs,
  fileBasicInfo,
  editingBasicInfo,
  needsBasicInfoStep,
  needsDomain,
  needsDocumentOutline,
  unchunkedFilesCount,
  setFileBasicInfo,
  setEditingBasicInfo,
  setDomain,
  setDocumentOutline,
  setJobStatus,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step3')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {isStep3Complete ? (
              <CheckCircleIcon color="var(--pf-global--success-color--100)" />
            ) : Object.values(fileBasicInfo).some(info => info.isComplete) ? (
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
              }}>3</span>
            )}
            Basic Information
            {!expandedSteps.step3 && Object.values(fileBasicInfo).some(info => info.isComplete) && (
              <Badge isRead style={{ marginLeft: '0.5rem' }}>
                {Object.values(fileBasicInfo).filter(info => info.isComplete).length} file{Object.values(fileBasicInfo).filter(info => info.isComplete).length !== 1 ? 's' : ''} configured
              </Badge>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Popover
              headerContent="Per-File Basic Information"
              bodyContent={`Configure ${[needsDomain && 'domain', needsDocumentOutline && 'document outline'].filter(Boolean).join(' and ')} for each chunked file. This information helps the model understand the context of your documents.`}
            >
              <Button variant="plain" aria-label="More info" onClick={(e) => e.stopPropagation()}>
                <OutlinedQuestionCircleIcon />
              </Button>
            </Popover>
            <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
              {expandedSteps.step3 ? '▼' : '▶'}
            </span>
          </div>
        </div>
      </CardTitle>
      {expandedSteps.step3 && <CardBody>
        {/* Warning about unchunked files */}
        {unchunkedFilesCount > 0 && (
          <Alert variant={AlertVariant.warning} isInline title={`${unchunkedFilesCount} file(s) not chunked`} style={{ marginBottom: '1rem' }}>
            Some files have not been chunked in Step 2. Go back to Step 2 to chunk them before adding basic information.
          </Alert>
        )}
        
        {/* Per-file basic info configuration */}
        {Object.entries(fileChunkConfigs)
          .filter(([_, config]) => config.isChunked)
          .map(([fileName, config], index) => {
            const savedInfo = fileBasicInfo[fileName] || { domain: '', documentOutline: '', isComplete: false };
            const isComplete = savedInfo.isComplete;
            const editingInfo = editingBasicInfo[fileName] || { domain: savedInfo.domain, documentOutline: savedInfo.documentOutline };
            
            const hasChanges = 
              (needsDomain && editingInfo.domain !== savedInfo.domain) || 
              (needsDocumentOutline && editingInfo.documentOutline !== savedInfo.documentOutline);
            
            const canSave = 
              (!needsDomain || editingInfo.domain) && 
              (!needsDocumentOutline || editingInfo.documentOutline);
            
            const fieldCount = (needsDomain ? 1 : 0) + (needsDocumentOutline ? 1 : 0);
            const fieldSpan = fieldCount === 2 ? 5 : 10;
            
            return (
              <div 
                key={fileName}
                style={{
                  padding: '1rem',
                  marginBottom: '0.75rem',
                  backgroundColor: isComplete ? '#f0fff0' : (hasChanges ? '#fffbf0' : '#fafafa'),
                  borderRadius: '6px',
                  border: isComplete ? '1px solid #3e8635' : (hasChanges ? '1px solid #f0ab00' : '1px solid #e0e0e0'),
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <Flex alignItems={{ default: 'alignItemsCenter' }}>
                    <FlexItem>
                      <FileIcon style={{ marginRight: '0.5rem', color: isComplete ? '#3e8635' : '#0066cc' }} />
                    </FlexItem>
                    <FlexItem>
                      <span style={{ fontWeight: '600' }}>{fileName}</span>
                      <Badge isRead style={{ marginLeft: '0.5rem' }}>{config.totalChunks} chunks</Badge>
                    </FlexItem>
                  </Flex>
                  <Flex style={{ gap: '0.5rem' }}>
                    {hasChanges && (
                      <Badge style={{ backgroundColor: '#fffbf0', color: '#f0ab00', border: '1px solid #f0ab00' }}>
                        Unsaved changes
                      </Badge>
                    )}
                    {isComplete && !hasChanges && (
                      <Badge style={{ backgroundColor: '#f0fff0', color: '#3e8635', border: '1px solid #3e8635' }}>
                        <CheckCircleIcon style={{ marginRight: '0.25rem' }} />
                        Saved
                      </Badge>
                    )}
                  </Flex>
                </div>
                
                <Grid hasGutter>
                  {/* Domain field */}
                  {needsDomain && (
                    <GridItem span={fieldSpan}>
                      <div>
                        <div style={{ fontWeight: 500, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          Domain
                          <Popover
                            headerContent="Domain Example"
                            bodyContent={
                              <div>
                                <p style={{ marginBottom: '0.5rem' }}>The subject area or category of your document. This helps the model generate contextually appropriate questions and answers relevant to the field.</p>
                                <div style={{ backgroundColor: '#f0f0f0', padding: '0.5rem 0.75rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}>
                                  <div><strong>Examples:</strong></div>
                                  <div style={{ marginTop: '0.25rem' }}>"Finance"</div>
                                  <div>"Computer Science"</div>
                                  <div>"Medical Research"</div>
                                  <div>"Legal Contracts"</div>
                                  <div>"articles/essays"</div>
                                </div>
                              </div>
                            }
                          >
                            <Button variant="plain" aria-label="Domain example" style={{ padding: 0, lineHeight: 1 }} onClick={(e) => e.stopPropagation()}>
                              <OutlinedQuestionCircleIcon style={{ fontSize: '0.85rem', color: '#0066cc' }} />
                            </Button>
                          </Popover>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                          Subject area of your document (e.g., Medical, Legal, Technical).
                        </div>
                        <TextInput
                          id={`domain-${index}`}
                          value={editingInfo.domain}
                          onChange={(_, value) => {
                            setEditingBasicInfo(prev => ({
                              ...prev,
                              [fileName]: {
                                ...prev[fileName],
                                domain: value,
                                documentOutline: prev[fileName]?.documentOutline ?? savedInfo.documentOutline,
                              }
                            }));
                          }}
                          placeholder="e.g., Finance, Computer Science, Medical Research"
                        />
                      </div>
                    </GridItem>
                  )}
                  {/* Document Outline field */}
                  {needsDocumentOutline && (
                    <GridItem span={fieldSpan}>
                      <div>
                        <div style={{ fontWeight: 500, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          Document Outline
                          <Popover
                            headerContent="Document Outline Example"
                            bodyContent={
                              <div>
                                <p style={{ marginBottom: '0.5rem' }}>A concise title or summary that accurately represents the entire document.</p>
                                <div style={{ backgroundColor: '#f0f0f0', padding: '0.5rem 0.75rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85rem' }}>
                                  <div><strong>Example:</strong></div>
                                  <div style={{ marginTop: '0.25rem' }}>"Annual Report - Financial Performance and Strategy Overview"</div>
                                </div>
                              </div>
                            }
                          >
                            <Button variant="plain" aria-label="Document outline example" style={{ padding: 0, lineHeight: 1 }} onClick={(e) => e.stopPropagation()}>
                              <OutlinedQuestionCircleIcon style={{ fontSize: '0.85rem', color: '#0066cc' }} />
                            </Button>
                          </Popover>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#6a6e73', marginBottom: '0.5rem' }}>
                          Brief description of the document's structure and main topics.
                        </div>
                        <TextInput
                          id={`document-outline-${index}`}
                          value={editingInfo.documentOutline}
                          onChange={(_, value) => {
                            setEditingBasicInfo(prev => ({
                              ...prev,
                              [fileName]: {
                                ...prev[fileName],
                                domain: prev[fileName]?.domain ?? savedInfo.domain,
                                documentOutline: value,
                              }
                            }));
                          }}
                          placeholder="e.g., The document contains excerpts from regulations on..."
                        />
                      </div>
                    </GridItem>
                  )}
                  <GridItem span={2} style={{ display: 'flex', alignItems: 'flex-end' }}>
                    <Button
                      variant="primary"
                      isSmall
                      onClick={() => {
                        if (canSave) {
                          setFileBasicInfo(prev => ({
                            ...prev,
                            [fileName]: {
                              domain: editingInfo.domain,
                              documentOutline: editingInfo.documentOutline,
                              isComplete: true,
                            }
                          }));
                          setEditingBasicInfo(prev => {
                            const newState = { ...prev };
                            delete newState[fileName];
                            return newState;
                          });
                          if (needsDomain) setDomain(editingInfo.domain);
                          if (needsDocumentOutline) setDocumentOutline(editingInfo.documentOutline);
                          setJobStatus('chunked');
                        }
                      }}
                      isDisabled={!canSave}
                      style={{ width: '100%' }}
                    >
                      {isComplete && !hasChanges ? 'Saved' : 'Save'}
                    </Button>
                  </GridItem>
                </Grid>
              </div>
            );
          })}
        
        {/* Summary */}
        {Object.values(fileBasicInfo).some(info => info.isComplete) && (
          <Alert 
            variant={AlertVariant.success} 
            isInline 
            title={`${Object.values(fileBasicInfo).filter(info => info.isComplete).length} of ${Object.values(fileChunkConfigs).filter(c => c.isChunked).length} files configured`}
            style={{ marginTop: '0.5rem' }}
          >
            Files with basic information configured are ready for dataset creation.
          </Alert>
        )}
      </CardBody>}
    </Card>
  );
};

export default BasicInfoSection;
