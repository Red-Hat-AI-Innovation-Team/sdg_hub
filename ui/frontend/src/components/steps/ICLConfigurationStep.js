import React, { useState, useEffect } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Form,
  FormGroup,
  TextInput,
  TextArea,
  Button,
  Alert,
  AlertVariant,
  Grid,
  GridItem,
  Radio,
  Badge,
  Tooltip,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
} from '@patternfly/react-core';
import { 
  CheckCircleIcon, 
  OutlinedQuestionCircleIcon,
  LightbulbIcon,
} from '@patternfly/react-icons';

/**
 * ICL Configuration Step Component
 * 
 * Standalone step for configuring In-Context Learning (ICL) templates.
 * This step helps users understand and configure ICL examples for better
 * generation quality.
 */
const ICLConfigurationStep = ({ 
  selectedFlow,
  iclConfig,
  onConfigChange,
  onError,
  availableTemplates = [],
}) => {
  // ICL mode: 'template' | 'custom'
  const [iclMode, setIclMode] = useState('template');
  
  // Template selection
  const [selectedTemplateIndex, setSelectedTemplateIndex] = useState(null);
  
  // Custom ICL fields
  const [customICL, setCustomICL] = useState({
    icl_document: '',
    icl_query_1: '',
    icl_response_1: '',
    icl_query_2: '',
    icl_response_2: '',
    icl_query_3: '',
    icl_response_3: '',
  });
  
  // Pre-fill from existing config
  useEffect(() => {
    if (iclConfig) {
      if (iclConfig.selectedTemplateIndex !== undefined) {
        setIclMode('template');
        setSelectedTemplateIndex(iclConfig.selectedTemplateIndex);
      } else if (iclConfig.icl_document) {
        setIclMode('custom');
        setCustomICL(iclConfig);
      }
    }
  }, [iclConfig]);
  
  // Determine if ICL is required for this flow
  const requiredColumns = selectedFlow?.dataset_requirements?.required_columns || [];
  const needsICL = requiredColumns.some(col => col.startsWith('icl_'));
  
  if (!needsICL) {
    return (
      <Card>
        <CardBody>
          <EmptyState>
            <EmptyStateIcon icon={CheckCircleIcon} color="green" />
            <Title headingLevel="h4" size="lg">ICL Not Required</Title>
            <EmptyStateBody>
              This flow does not require In-Context Learning (ICL) examples.
              You can proceed to the next step.
            </EmptyStateBody>
          </EmptyState>
        </CardBody>
      </Card>
    );
  }

  const handleSaveConfig = () => {
    if (iclMode === 'template' && selectedTemplateIndex !== null) {
      onConfigChange({
        selectedTemplateIndex,
        ...availableTemplates[selectedTemplateIndex],
      });
    } else if (iclMode === 'custom') {
      onConfigChange(customICL);
    }
  };

  return (
    <Grid hasGutter>
      {/* Left Panel - Configuration */}
      <GridItem span={7}>
        <Card>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              In-Context Learning (ICL) Configuration
            </Title>
          </CardTitle>
          <CardBody>
            {/* Explanation Alert */}
            <Alert
              variant={AlertVariant.info}
              title="What is ICL?"
              isInline
              style={{ marginBottom: '1.5rem' }}
            >
              <p>
                In-Context Learning provides example question-answer pairs that help the model 
                understand the expected format and style. Good examples significantly improve 
                generation quality.
              </p>
            </Alert>
            
            {/* Mode Selection */}
            <FormGroup label="Configuration Method" fieldId="icl-mode" style={{ marginBottom: '1.5rem' }}>
              <Radio
                isChecked={iclMode === 'template'}
                name="icl-mode"
                onChange={() => setIclMode('template')}
                label="Use a pre-built template"
                id="icl-mode-template"
                description="Select from curated ICL templates"
              />
              <Radio
                isChecked={iclMode === 'custom'}
                name="icl-mode"
                onChange={() => setIclMode('custom')}
                label="Create custom ICL examples"
                id="icl-mode-custom"
                description="Write your own ICL document and Q&A pairs"
                style={{ marginTop: '0.5rem' }}
              />
            </FormGroup>
            
            {/* Template Selection */}
            {iclMode === 'template' && (
              <div>
                <Title headingLevel="h4" size="md" style={{ marginBottom: '1rem' }}>
                  Select a Template
                </Title>
                {availableTemplates.length === 0 ? (
                  <Alert variant={AlertVariant.warning} isInline title="No templates available">
                    No pre-built templates are available for this flow. Please create custom ICL examples.
                  </Alert>
                ) : (
                  <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                    {availableTemplates.map((template, idx) => (
                      <Card
                        key={idx}
                        isSelectable
                        isSelected={selectedTemplateIndex === idx}
                        onClick={() => setSelectedTemplateIndex(idx)}
                        style={{ marginBottom: '1rem', cursor: 'pointer' }}
                      >
                        <CardBody>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Title headingLevel="h5" size="md">
                              {template.name || `Template ${idx + 1}`}
                            </Title>
                            {selectedTemplateIndex === idx && (
                              <CheckCircleIcon color="green" />
                            )}
                          </div>
                          <p style={{ 
                            fontSize: '0.875rem', 
                            color: '#6a6e73',
                            marginTop: '0.5rem',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '100px',
                            overflow: 'hidden',
                          }}>
                            {template.description || template.icl_document?.substring(0, 200)}...
                          </p>
                        </CardBody>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {/* Custom ICL Form */}
            {iclMode === 'custom' && (
              <Form>
                <FormGroup
                  label={
                    <Tooltip
                      content={
                        <div style={{ maxWidth: '300px' }}>
                          <strong>ICL Document</strong>
                          <p style={{ marginTop: '0.5rem' }}>
                            A sample document used in the ICL template to show the model what kind of source material it will be working with.
                          </p>
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
                        ICL Document
                        <OutlinedQuestionCircleIcon style={{ fontSize: '0.875rem', color: '#6a6e73' }} />
                      </span>
                    </Tooltip>
                  }
                  isRequired
                  fieldId="icl-document"
                  helperText="A sample document that represents your data"
                >
                  <TextArea
                    isRequired
                    id="icl-document"
                    value={customICL.icl_document}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_document: val }))}
                    placeholder="Enter a sample document..."
                    rows={6}
                  />
                </FormGroup>
                
                {/* Q&A Pair 1 */}
                <Title headingLevel="h4" size="md" style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>
                  Example Q&A Pair 1 (Required)
                </Title>
                <FormGroup label="Question 1" isRequired fieldId="icl-query-1">
                  <TextInput
                    isRequired
                    type="text"
                    id="icl-query-1"
                    value={customICL.icl_query_1}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_query_1: val }))}
                    placeholder="Enter an example question..."
                  />
                </FormGroup>
                <FormGroup label="Answer 1" isRequired fieldId="icl-response-1">
                  <TextArea
                    isRequired
                    id="icl-response-1"
                    value={customICL.icl_response_1}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_response_1: val }))}
                    placeholder="Enter the expected answer..."
                    rows={3}
                  />
                </FormGroup>
                
                {/* Q&A Pair 2 */}
                <Title headingLevel="h4" size="md" style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>
                  Example Q&A Pair 2 (Optional)
                </Title>
                <FormGroup label="Question 2" fieldId="icl-query-2">
                  <TextInput
                    type="text"
                    id="icl-query-2"
                    value={customICL.icl_query_2}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_query_2: val }))}
                    placeholder="Enter another example question..."
                  />
                </FormGroup>
                <FormGroup label="Answer 2" fieldId="icl-response-2">
                  <TextArea
                    id="icl-response-2"
                    value={customICL.icl_response_2}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_response_2: val }))}
                    placeholder="Enter the expected answer..."
                    rows={3}
                  />
                </FormGroup>
                
                {/* Q&A Pair 3 */}
                <Title headingLevel="h4" size="md" style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>
                  Example Q&A Pair 3 (Optional)
                </Title>
                <FormGroup label="Question 3" fieldId="icl-query-3">
                  <TextInput
                    type="text"
                    id="icl-query-3"
                    value={customICL.icl_query_3}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_query_3: val }))}
                    placeholder="Enter another example question..."
                  />
                </FormGroup>
                <FormGroup label="Answer 3" fieldId="icl-response-3">
                  <TextArea
                    id="icl-response-3"
                    value={customICL.icl_response_3}
                    onChange={(e, val) => setCustomICL(prev => ({ ...prev, icl_response_3: val }))}
                    placeholder="Enter the expected answer..."
                    rows={3}
                  />
                </FormGroup>
              </Form>
            )}
            
            {/* Save Button */}
            <Button
              variant="primary"
              onClick={handleSaveConfig}
              style={{ marginTop: '1.5rem' }}
              isDisabled={
                (iclMode === 'template' && selectedTemplateIndex === null) ||
                (iclMode === 'custom' && (!customICL.icl_document || !customICL.icl_query_1 || !customICL.icl_response_1))
              }
            >
              Save ICL Configuration
            </Button>
          </CardBody>
        </Card>
      </GridItem>
      
      {/* Right Panel - Help & Tips */}
      <GridItem span={5}>
        <Card>
          <CardTitle>
            <Title headingLevel="h3" size="lg">
              <LightbulbIcon style={{ marginRight: '0.5rem' }} />
              Tips for Good ICL Examples
            </Title>
          </CardTitle>
          <CardBody>
            <div style={{ fontSize: '0.9rem' }}>
              <p style={{ marginBottom: '1rem' }}>
                <strong>1. Use Representative Documents</strong><br />
                Your ICL document should be similar in style and content to your actual data.
              </p>
              <p style={{ marginBottom: '1rem' }}>
                <strong>2. Show Diverse Questions</strong><br />
                Include different types of questions (factual, analytical, comparative) to guide the model.
              </p>
              <p style={{ marginBottom: '1rem' }}>
                <strong>3. Provide Detailed Answers</strong><br />
                Answers should match the format and detail level you want in your generated data.
              </p>
              <p style={{ marginBottom: '1rem' }}>
                <strong>4. Be Consistent</strong><br />
                Maintain consistent formatting across all Q&A pairs.
              </p>
              <p style={{ marginBottom: '0' }}>
                <strong>5. Quality Over Quantity</strong><br />
                A few well-crafted examples are better than many poor ones.
              </p>
            </div>
          </CardBody>
        </Card>
        
        {/* Example Preview Card */}
        {iclMode === 'custom' && customICL.icl_document && (
          <Card style={{ marginTop: '1rem' }}>
            <CardTitle>
              <Title headingLevel="h4" size="md">
                Preview
              </Title>
            </CardTitle>
            <CardBody>
              <div style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '1rem', 
                borderRadius: '4px',
                fontSize: '0.85rem'
              }}>
                <strong>Document:</strong>
                <p style={{ 
                  marginTop: '0.5rem',
                  maxHeight: '100px',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap'
                }}>
                  {customICL.icl_document.substring(0, 300)}
                  {customICL.icl_document.length > 300 && '...'}
                </p>
                
                {customICL.icl_query_1 && (
                  <>
                    <strong style={{ marginTop: '1rem', display: 'block' }}>Q1:</strong>
                    <p style={{ margin: '0.25rem 0' }}>{customICL.icl_query_1}</p>
                    {customICL.icl_response_1 && (
                      <>
                        <strong>A1:</strong>
                        <p style={{ margin: '0.25rem 0', color: '#3e8635' }}>
                          {customICL.icl_response_1.substring(0, 100)}
                          {customICL.icl_response_1.length > 100 && '...'}
                        </p>
                      </>
                    )}
                  </>
                )}
              </div>
            </CardBody>
          </Card>
        )}
      </GridItem>
    </Grid>
  );
};

export default ICLConfigurationStep;
