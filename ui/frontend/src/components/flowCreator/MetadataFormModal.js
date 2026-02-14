import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Form,
  FormGroup,
  TextInput,
  TextArea,
  Alert,
  AlertVariant,
  Grid,
  GridItem,
  Chip,
  ChipGroup,
} from '@patternfly/react-core';

/**
 * Metadata Form Modal
 * 
 * Collects flow metadata before saving:
 * - Name (required)
 * - Description (required)
 * - Version
 * - Author
 * - Tags
 * - License
 */
const MetadataFormModal = ({ initialMetadata, onSubmit, onClose }) => {
  const [metadata, setMetadata] = useState({
    name: '',
    description: '',
    version: '1.0.0',
    author: 'SDG Hub User',
    tags: [],
    required_columns: [],
    license: 'Apache-2.0',
    ...initialMetadata
  });
  const [tagInput, setTagInput] = useState('');
  const [columnInput, setColumnInput] = useState('');
  const [errors, setErrors] = useState({});

  /**
   * Handle field change
   */
  const handleChange = (field, value) => {
    setMetadata(prev => ({ ...prev, [field]: value }));
    // Clear error for this field
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  /**
   * Add tag
   */
  const handleAddTag = () => {
    if (tagInput.trim()) {
      setMetadata(prev => ({
        ...prev,
        tags: [...(prev.tags || []), tagInput.trim()]
      }));
      setTagInput('');
    }
  };

  /**
   * Remove tag
   */
  const handleRemoveTag = (tagToRemove) => {
    setMetadata(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  /**
   * Add required column
   */
  const handleAddColumn = () => {
    if (columnInput.trim()) {
      setMetadata(prev => ({
        ...prev,
        required_columns: [...(prev.required_columns || []), columnInput.trim()]
      }));
      setColumnInput('');
    }
  };

  /**
   * Remove required column
   */
  const handleRemoveColumn = (columnToRemove) => {
    setMetadata(prev => ({
      ...prev,
      required_columns: (prev.required_columns || []).filter(col => col !== columnToRemove)
    }));
  };

  /**
   * Handle column input key press (Enter to add)
   */
  const handleColumnKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAddColumn();
    }
  };

  /**
   * Validate form
   */
  const validate = () => {
    const newErrors = {};
    
    if (!metadata.name || metadata.name.trim() === '') {
      newErrors.name = 'Flow name is required';
    }
    
    if (!metadata.description || metadata.description.trim() === '') {
      newErrors.description = 'Description is required';
    }
    
    if (!metadata.version || metadata.version.trim() === '') {
      newErrors.version = 'Version is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handle submit
   */
  const handleSubmit = () => {
    if (validate()) {
      onSubmit(metadata);
    }
  };

  /**
   * Handle tag input key press (Enter to add)
   */
  const handleTagKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAddTag();
    }
  };

  return (
    <Modal
      variant={ModalVariant.medium}
      title="Flow Metadata"
      isOpen={true}
      onClose={onClose}
      actions={[
        <Button key="save" variant="primary" onClick={handleSubmit}>
          Save Flow
        </Button>,
        <Button key="cancel" variant="link" onClick={onClose}>
          Cancel
        </Button>,
      ]}
    >
      <Alert
        variant={AlertVariant.info}
        isInline
        title="Complete your flow metadata"
        style={{ marginBottom: '1.5rem' }}
      >
        Provide information about your flow before saving. This metadata helps users understand what your flow does.
      </Alert>

      <Form>
        {/* Flow Name */}
        <FormGroup
          label="Flow Name"
          isRequired
          fieldId="flow-name"
          helperText="A descriptive name for your flow"
          validated={errors.name ? 'error' : 'default'}
          helperTextInvalid={errors.name}
        >
          <TextInput
            isRequired
            type="text"
            id="flow-name"
            value={metadata.name}
            onChange={(event, value) => handleChange('name', value)}
            validated={errors.name ? 'error' : 'default'}
            placeholder="e.g., My Custom QA Generation Flow"
          />
        </FormGroup>

        {/* Description */}
        <FormGroup
          label="Description"
          isRequired
          fieldId="description"
          helperText="Explain what this flow does and when to use it"
          validated={errors.description ? 'error' : 'default'}
          helperTextInvalid={errors.description}
        >
          <TextArea
            isRequired
            id="description"
            value={metadata.description}
            onChange={(event, value) => handleChange('description', value)}
            validated={errors.description ? 'error' : 'default'}
            rows={4}
            placeholder="This flow generates question-answer pairs from documents using..."
          />
        </FormGroup>

        {/* Required Columns */}
        <FormGroup
          label="Required Columns"
          fieldId="required-columns"
          helperText="Specify dataset columns required by your flow (e.g., 'document', 'context'). Leave empty to skip column validation during dataset upload."
        >
          <div style={{ marginBottom: '0.5rem' }}>
            <TextInput
              type="text"
              id="column-input"
              value={columnInput}
              onChange={(event, value) => setColumnInput(value)}
              onKeyPress={handleColumnKeyPress}
              placeholder="e.g., document"
              style={{ maxWidth: '400px' }}
            />
            <Button 
              variant="secondary" 
              onClick={handleAddColumn}
              style={{ marginLeft: '0.5rem' }}
            >
              Add Column
            </Button>
          </div>
          {metadata.required_columns && metadata.required_columns.length > 0 && (
            <ChipGroup categoryName="Required Columns">
              {metadata.required_columns.map(col => (
                <Chip key={col} onClick={() => handleRemoveColumn(col)}>
                  {col}
                </Chip>
              ))}
            </ChipGroup>
          )}
          {(!metadata.required_columns || metadata.required_columns.length === 0) && (
            <Alert
              variant={AlertVariant.info}
              isInline
              isPlain
              title="No required columns specified - dataset column validation will be skipped"
              style={{ marginTop: '0.5rem' }}
            />
          )}
        </FormGroup>

        <Grid hasGutter>
          {/* Version */}
          <GridItem span={6}>
            <FormGroup
              label="Version"
              isRequired
              fieldId="version"
              validated={errors.version ? 'error' : 'default'}
              helperTextInvalid={errors.version}
            >
              <TextInput
                isRequired
                type="text"
                id="version"
                value={metadata.version}
                onChange={(event, value) => handleChange('version', value)}
                validated={errors.version ? 'error' : 'default'}
                placeholder="1.0.0"
              />
            </FormGroup>
          </GridItem>

          {/* Author */}
          <GridItem span={6}>
            <FormGroup label="Author" fieldId="author">
              <TextInput
                type="text"
                id="author"
                value={metadata.author}
                onChange={(event, value) => handleChange('author', value)}
                placeholder="Your Name"
              />
            </FormGroup>
          </GridItem>
        </Grid>

        {/* Tags */}
        <FormGroup
          label="Tags"
          fieldId="tags"
          helperText="Add tags to categorize your flow (press Enter to add)"
        >
          <div style={{ marginBottom: '0.5rem' }}>
            <TextInput
              type="text"
              id="tag-input"
              value={tagInput}
              onChange={(event, value) => setTagInput(value)}
              onKeyPress={handleTagKeyPress}
              placeholder="e.g., question-generation, custom"
              style={{ maxWidth: '400px' }}
            />
            <Button 
              variant="secondary" 
              onClick={handleAddTag}
              style={{ marginLeft: '0.5rem' }}
            >
              Add Tag
            </Button>
          </div>
          {metadata.tags && metadata.tags.length > 0 && (
            <ChipGroup categoryName="Tags">
              {metadata.tags.map(tag => (
                <Chip key={tag} onClick={() => handleRemoveTag(tag)}>
                  {tag}
                </Chip>
              ))}
            </ChipGroup>
          )}
        </FormGroup>

        {/* License */}
        <FormGroup label="License" fieldId="license">
          <TextInput
            type="text"
            id="license"
            value={metadata.license}
            onChange={(event, value) => handleChange('license', value)}
            placeholder="Apache-2.0"
          />
        </FormGroup>
      </Form>
    </Modal>
  );
};

export default MetadataFormModal;

