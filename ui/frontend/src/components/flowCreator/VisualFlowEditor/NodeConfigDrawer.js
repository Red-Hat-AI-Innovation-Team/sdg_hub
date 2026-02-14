import React, { useState, useEffect } from 'react';
import {
  DrawerPanelContent,
  DrawerHead,
  DrawerActions,
  DrawerCloseButton,
  DrawerPanelBody,
  Title,
  Button,
  Form,
  FormGroup,
  TextInput,
  TextArea,
  NumberInput,
  Checkbox,
  Select,
  SelectOption,
  SelectList,
  MenuToggle,
  FormHelperText,
  HelperText,
  HelperTextItem,
  ExpandableSection,
  Label,
  LabelGroup,
  Alert,
  AlertVariant,
  ActionGroup,
  Divider,
  Badge,
  Split,
  SplitItem,
  Modal,
  ModalVariant,
  Popover,
} from '@patternfly/react-core';
import { 
  TrashIcon, 
  SaveIcon, 
  PlusCircleIcon,
  TimesIcon,
  ExclamationCircleIcon,
  EditIcon,
  OutlinedQuestionCircleIcon,
} from '@patternfly/react-icons';

import { NODE_TYPE_CONFIG, NODE_TYPES } from './constants';
import { getNodeConfig } from './nodes';
import PromptEditorModal from '../PromptEditorModal';
import { promptAPI, workspaceAPI } from '../../../services/api';

/**
 * Node Configuration Drawer Component
 * 
 * Side drawer that appears when a node is selected.
 * Displays form fields based on the node type configuration.
 */
const NodeConfigDrawer = ({ node, onClose, onSave, onDelete, existingFlowName, workspaceId }) => {
  const [config, setConfig] = useState({});
  const [errors, setErrors] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectOpen, setSelectOpen] = useState({});
  const [tagInputs, setTagInputs] = useState({});
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [tempFlowName, setTempFlowName] = useState(null);

  const nodeTypeConfig = NODE_TYPE_CONFIG[node.type];
  const nodeConfig = getNodeConfig(node.type);

  // Check if this node type requires prompt editing
  const needsPromptEditor = node.type === NODE_TYPES.PROMPT || node.type === NODE_TYPES.EVAL;

  /**
   * Initialize config from node - use JSON comparison to avoid infinite loops
   */
  const nodeConfigJson = JSON.stringify(node?.config || {});
  useEffect(() => {
    if (node?.config) {
      setConfig({ ...nodeConfig?.defaultConfig, ...node.config });
    } else if (nodeConfig?.defaultConfig) {
      setConfig({ ...nodeConfig.defaultConfig });
    }
    setErrors({});
  }, [node?.id, nodeConfigJson, nodeConfig?.defaultConfig]);

  /**
   * Handle field change
   */
  const handleChange = (fieldName, value) => {
    setConfig(prev => ({
      ...prev,
      [fieldName]: value,
    }));
    // Clear error for this field
    if (errors[fieldName]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[fieldName];
        return newErrors;
      });
    }
  };

  /**
   * Handle tag addition (for array fields)
   * If the field has autoAddToUserMessage flag, also append {{tag}} to user_message
   */
  const handleAddTag = (fieldName, value, fieldConfig = null) => {
    if (!value?.trim()) return;
    
    const trimmedValue = value.trim();
    const currentTags = config[fieldName] || [];
    
    if (!currentTags.includes(trimmedValue)) {
      handleChange(fieldName, [...currentTags, trimmedValue]);
      
      // If this field has autoAddToUserMessage flag, append {{variable}} to user_message
      if (fieldConfig?.autoAddToUserMessage) {
        const variableTemplate = `{{${trimmedValue}}}`;
        const currentUserMessage = config.user_message || '';
        
        // Check if the variable is already in the user message
        if (!currentUserMessage.includes(variableTemplate)) {
          // Add the variable to the end of the user message
          const newUserMessage = currentUserMessage.trim() 
            ? `${currentUserMessage.trim()} ${variableTemplate}`
            : variableTemplate;
          handleChange('user_message', newUserMessage);
        }
      }
    }
    setTagInputs(prev => ({ ...prev, [fieldName]: '' }));
  };

  /**
   * Handle tag removal
   */
  const handleRemoveTag = (fieldName, tagToRemove) => {
    const currentTags = config[fieldName] || [];
    handleChange(fieldName, currentTags.filter(tag => tag !== tagToRemove));
  };

  /**
   * Handle key-value pair change (for mapping fields)
   */
  const handleKeyValueChange = (fieldName, key, value, isKey = false) => {
    const mapping = { ...(config[fieldName] || {}) };
    
    if (isKey) {
      // Changing the key - need to remove old and add new
      const oldValue = mapping[key];
      delete mapping[key];
      mapping[value] = oldValue || '';
    } else {
      mapping[key] = value;
    }
    
    handleChange(fieldName, mapping);
  };

  /**
   * Handle adding a new key-value pair
   */
  const handleAddKeyValue = (fieldName) => {
    const mapping = { ...(config[fieldName] || {}) };
    const newKey = `column_${Object.keys(mapping).length + 1}`;
    mapping[newKey] = '';
    handleChange(fieldName, mapping);
  };

  /**
   * Handle removing a key-value pair
   */
  const handleRemoveKeyValue = (fieldName, keyToRemove) => {
    const mapping = { ...(config[fieldName] || {}) };
    delete mapping[keyToRemove];
    handleChange(fieldName, mapping);
  };

  /**
   * Handle prompt editor save
   */
  const handlePromptSave = async (promptData, promptName) => {
    try {
      // If we have a workspace, save directly to workspace
      if (workspaceId) {
        // Use existing prompt_config_path if available (to overwrite the original file)
        // Otherwise use the promptName (block_name)
        const promptFilename = config.prompt_config_path || `${promptName}.yaml`;
        const promptConfig = {
          messages: promptData,
        };
        
        // Save prompt to workspace
        const result = await workspaceAPI.updatePrompt(workspaceId, promptFilename, promptConfig);
        console.log('Prompt saved to workspace:', result.full_prompt_path, '(filename:', promptFilename, ')');

        // Create updated config
        const updatedConfig = {
          ...config,
          prompt_config_path: result.prompt_filename,
          system_message: promptData.find(m => m.role === 'system')?.content || '',
          user_message: promptData.find(m => m.role === 'user')?.content || '',
        };

        // Update local state
        setConfig(updatedConfig);

        // Also trigger save to update flow.yaml in workspace
        // This ensures the test uses the updated prompt reference
        if (onSave) {
          onSave(updatedConfig);
        }

        setShowPromptEditor(false);
        return;
      }

      // Fallback: Use existing flow name if available, otherwise generate temp name
      const flowNameToUse = existingFlowName || tempFlowName || `temp_visual_flow_${Date.now()}`;
      
      if (!tempFlowName && !existingFlowName) {
        setTempFlowName(flowNameToUse);
      }

      // Save prompt to server (old method)
      const response = await promptAPI.savePrompt({
        prompt_name: promptName,
        prompt_content: promptData,
        flow_name: flowNameToUse.replace(' (Custom)', '').replace(' (Copy)', ''),
      });

      // Update config with prompt file path
      setConfig(prev => ({
        ...prev,
        prompt_config_path: response.prompt_filename,
        system_message: promptData.find(m => m.role === 'system')?.content || '',
        user_message: promptData.find(m => m.role === 'user')?.content || '',
      }));

      setShowPromptEditor(false);
    } catch (error) {
      console.error('Failed to save prompt:', error);
      alert('Failed to save prompt: ' + (error.message || error));
    }
  };

  /**
   * Open prompt editor
   */
  const handleOpenPromptEditor = () => {
    // Ensure we have a block name for the prompt file
    if (!config.block_name?.trim()) {
      setErrors({ block_name: 'Please enter a block name first' });
      return;
    }
    setShowPromptEditor(true);
  };

  /**
   * Validate and save
   */
  const handleSave = () => {
    // Run validation
    if (nodeConfig?.validate) {
      const validationErrors = nodeConfig.validate(config);
      if (Object.keys(validationErrors).length > 0) {
        setErrors(validationErrors);
        return;
      }
    }

    onSave(config);
  };

  /**
   * Render a form field based on its type
   */
  const renderField = (field) => {
    // Check showWhen condition
    if (field.showWhen && !field.showWhen(config)) {
      return null;
    }

    const value = config[field.name];
    const error = errors[field.name];
    const validated = error ? 'error' : 'default';

    switch (field.type) {
      case 'text':
        return (
          <FormGroup
            key={field.name}
            label={field.label}
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
            data-tour={field.name === 'input_cols' ? 'input-column-field' : undefined}
          >
            <TextInput
              id={field.name}
              value={value || ''}
              onChange={(event, val) => handleChange(field.name, val)}
              placeholder={field.placeholder}
              validated={validated}
            />
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem variant={error ? 'error' : 'default'} style={{ fontSize: '11px' }}>
                    {error || field.helperText}
                  </HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'textarea':
        // Use compact rows for system message, slightly more for user message
        const compactRows = field.name === 'system_message' ? 2 : Math.min(field.rows || 4, 5);
        return (
          <FormGroup
            key={field.name}
            label={
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {field.label}
                {field.hasInfoTooltip && field.infoTooltipContent && (
                  <Popover
                    headerContent="Prompt Template Tips"
                    bodyContent={
                      <div style={{ 
                        whiteSpace: 'pre-wrap', 
                        fontSize: '12px', 
                        lineHeight: '1.5',
                        maxWidth: '400px',
                      }}>
                        {field.infoTooltipContent.split('```').map((part, idx) => {
                          if (idx % 2 === 1) {
                            // Code block
                            return (
                              <pre key={idx} style={{
                                background: '#f5f5f5',
                                padding: '8px',
                                borderRadius: '4px',
                                fontSize: '11px',
                                overflow: 'auto',
                                margin: '8px 0',
                              }}>
                                {part.trim()}
                              </pre>
                            );
                          }
                          // Regular text - handle **bold** markdown
                          return (
                            <span key={idx}>
                              {part.split(/\*\*(.*?)\*\*/).map((text, i) => 
                                i % 2 === 1 
                                  ? <strong key={i}>{text}</strong> 
                                  : text
                              )}
                            </span>
                          );
                        })}
                      </div>
                    }
                    position="right"
                    maxWidth="450px"
                  >
                    <button
                      type="button"
                      aria-label="More info"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: '2px',
                        cursor: 'pointer',
                        color: '#0066cc',
                        display: 'flex',
                        alignItems: 'center',
                      }}
                    >
                      <OutlinedQuestionCircleIcon />
                    </button>
                  </Popover>
                )}
              </span>
            }
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
            data-tour={field.name === 'system_message' ? 'system-message' : (field.name === 'user_message' ? 'user-message' : undefined)}
          >
            <TextArea
              id={field.name}
              value={value || ''}
              onChange={(event, val) => handleChange(field.name, val)}
              placeholder={field.placeholder}
              rows={compactRows}
              validated={validated}
              style={{ fontFamily: 'monospace', fontSize: '12px' }}
              resizeOrientation="vertical"
            />
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem variant={error ? 'error' : 'default'} style={{ fontSize: '11px' }}>
                    {error || field.helperText}
                  </HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'number':
        return (
          <FormGroup
            key={field.name}
            label={field.label}
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
          >
            <NumberInput
              id={field.name}
              value={value ?? field.defaultValue ?? 0}
              onMinus={() => handleChange(field.name, Math.max(field.min ?? 0, (value ?? field.defaultValue ?? 0) - (field.step || 1)))}
              onPlus={() => handleChange(field.name, Math.min(field.max ?? Infinity, (value ?? field.defaultValue ?? 0) + (field.step || 1)))}
              onChange={(event) => {
                const val = parseFloat(event.target.value);
                handleChange(field.name, isNaN(val) ? field.defaultValue : val);
              }}
              min={field.min}
              max={field.max}
              widthChars={8}
            />
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem style={{ fontSize: '11px' }}>{field.helperText}</HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'slider':
        const sliderValue = value ?? field.defaultValue ?? 0;
        const sliderStep = field.step || 0.1;
        const sliderMin = field.min ?? 0;
        const sliderMax = field.max ?? 1;

        return (
          <FormGroup
            key={field.name}
            label={
              <span>
                {field.label}
                <span style={{ 
                  marginLeft: '6px', 
                  fontWeight: 'bold',
                  color: '#0066cc',
                  background: '#e7f1fa',
                  padding: '1px 6px',
                  borderRadius: '4px',
                  fontSize: '12px',
                }}>
                  {sliderValue}
                </span>
              </span>
            }
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
          >
            <div style={{ padding: '4px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '11px', color: '#6a6e73' }}>{sliderMin}</span>
                <div style={{ flex: 1 }}>
                  <input
                    type="range"
                    id={field.name}
                    value={sliderValue}
                    min={sliderMin}
                    max={sliderMax}
                    step={sliderStep}
                    onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                    style={{
                      width: '100%',
                      height: '5px',
                      cursor: 'pointer',
                      accentColor: '#0066cc',
                    }}
                  />
                </div>
                <span style={{ fontSize: '11px', color: '#6a6e73' }}>{sliderMax}</span>
              </div>
            </div>
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem style={{ fontSize: '11px' }}>{field.helperText}</HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'checkbox':
        return (
          <FormGroup key={field.name} fieldId={field.name} style={{ marginBottom: '8px' }}>
            <Checkbox
              id={field.name}
              label={field.label}
              isChecked={value ?? field.defaultValue ?? false}
              onChange={(event, checked) => handleChange(field.name, checked)}
            />
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem style={{ fontSize: '11px' }}>{field.helperText}</HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'select':
        return (
          <FormGroup
            key={field.name}
            label={field.label}
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
          >
            <Select
              id={field.name}
              isOpen={selectOpen[field.name] || false}
              selected={value || field.defaultValue}
              onSelect={(event, selection) => {
                handleChange(field.name, selection);
                setSelectOpen(prev => ({ ...prev, [field.name]: false }));
              }}
              onOpenChange={(isOpen) => setSelectOpen(prev => ({ ...prev, [field.name]: isOpen }))}
              toggle={(toggleRef) => (
                <MenuToggle 
                  ref={toggleRef} 
                  onClick={() => setSelectOpen(prev => ({ ...prev, [field.name]: !prev[field.name] }))}
                  isFullWidth
                >
                  {field.options?.find(o => o.value === (value || field.defaultValue))?.label || value || field.defaultValue}
                </MenuToggle>
              )}
            >
              <SelectList>
                {field.options?.map(option => (
                  <SelectOption key={option.value} value={option.value}>
                    {option.label}
                  </SelectOption>
                ))}
              </SelectList>
            </Select>
            {field.helperText && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem style={{ fontSize: '11px' }}>{field.helperText}</HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'tags':
        // Ensure tags is always an array
        const tags = Array.isArray(value) ? value : (value ? [value] : []);
        const tagInput = tagInputs[field.name] || '';
        return (
          <FormGroup
            key={field.name}
            label={field.label}
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
            data-tour={field.name === 'start_tags' ? 'start-tags-field' : undefined}
          >
            <div style={{ marginBottom: '6px' }}>
              <LabelGroup>
                {tags.map((tag, index) => (
                  <Label
                    key={index}
                    color="blue"
                    onClose={() => handleRemoveTag(field.name, tag)}
                    isCompact
                  >
                    {tag}
                  </Label>
                ))}
              </LabelGroup>
            </div>
            <Split hasGutter>
              <SplitItem isFilled>
                <TextInput
                  id={`${field.name}-input`}
                  value={tagInput}
                  onChange={(event, val) => setTagInputs(prev => ({ ...prev, [field.name]: val }))}
                  placeholder={field.placeholder}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddTag(field.name, tagInput, field);
                    }
                  }}
                  list={field.suggestions ? `${field.name}-suggestions` : undefined}
                />
                {field.suggestions && (
                  <datalist id={`${field.name}-suggestions`}>
                    {field.suggestions.filter(s => !tags.includes(s)).map(s => (
                      <option key={s} value={s} />
                    ))}
                  </datalist>
                )}
              </SplitItem>
              <SplitItem>
                <Button
                  variant="secondary"
                  onClick={() => handleAddTag(field.name, tagInput, field)}
                  size="sm"
                >
                  Add
                </Button>
              </SplitItem>
            </Split>
            {(error || field.helperText) && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem variant={error ? 'error' : 'default'} style={{ fontSize: '11px' }}>
                    {error || field.helperText}
                  </HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      case 'keyvalue':
        const mapping = value || {};
        return (
          <FormGroup
            key={field.name}
            label={field.label}
            isRequired={field.required}
            fieldId={field.name}
            style={{ marginBottom: '10px' }}
          >
            {Object.entries(mapping).map(([key, val], index) => (
              <Split key={index} hasGutter style={{ marginBottom: '6px' }}>
                <SplitItem isFilled>
                  <TextInput
                    value={key}
                    onChange={(event, newKey) => handleKeyValueChange(field.name, key, newKey, true)}
                    placeholder={field.keyPlaceholder || 'Key'}
                    aria-label="Key"
                  />
                </SplitItem>
                <SplitItem style={{ alignSelf: 'center', fontSize: '12px' }}>→</SplitItem>
                <SplitItem isFilled>
                  <TextInput
                    value={val}
                    onChange={(event, newVal) => handleKeyValueChange(field.name, key, newVal, false)}
                    placeholder={field.valuePlaceholder || 'Value'}
                    aria-label="Value"
                  />
                </SplitItem>
                <SplitItem>
                  <Button
                    variant="plain"
                    icon={<TimesIcon />}
                    onClick={() => handleRemoveKeyValue(field.name, key)}
                    aria-label="Remove"
                    style={{ padding: '4px' }}
                  />
                </SplitItem>
              </Split>
            ))}
            <Button
              variant="link"
              icon={<PlusCircleIcon />}
              onClick={() => handleAddKeyValue(field.name)}
              size="sm"
            >
              Add mapping
            </Button>
            {(error || field.helperText) && (
              <FormHelperText>
                <HelperText>
                  <HelperTextItem variant={error ? 'error' : 'default'} style={{ fontSize: '11px' }}>
                    {error || field.helperText}
                  </HelperTextItem>
                </HelperText>
              </FormHelperText>
            )}
          </FormGroup>
        );

      default:
        return null;
    }
  };

  if (!node || !nodeConfig) {
    return null;
  }

  // Separate basic and advanced fields
  const basicFields = nodeConfig.fields?.filter(f => !f.advanced) || [];
  const advancedFields = nodeConfig.fields?.filter(f => f.advanced) || [];

  return (
    <div
      style={{
        width: '400px',
        height: '100%',
        background: '#fff',
        borderLeft: '1px solid #d2d2d2',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        flexShrink: 0,
      }}
      className="compact-config-drawer"
    >
      {/* Header - Compact */}
      <div style={{ 
        padding: '10px 12px', 
        borderBottom: '1px solid #d2d2d2',
        background: `${nodeTypeConfig.color}10`,
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
              {nodeTypeConfig.icon && React.createElement(nodeTypeConfig.icon, { 
                style: { color: nodeTypeConfig.color, fontSize: '18px' } 
              })}
              <Title headingLevel="h4" size="md">
                {config.block_name || nodeTypeConfig.label}
              </Title>
            </div>
            <div style={{ color: '#6a6e73', fontSize: '12px' }}>
              {nodeTypeConfig.description}
            </div>
          </div>
          <Button variant="plain" onClick={onClose} aria-label="Close drawer" style={{ padding: '4px' }}>
            <TimesIcon />
          </Button>
        </div>
      </div>

      {/* Form Content - Scrollable with footer always visible */}
      <div style={{ flex: 1, overflow: 'auto', padding: '10px 12px', minHeight: 0 }}>
        {/* Validation Errors Summary */}
        {Object.keys(errors).length > 0 && (
          <Alert
            variant={AlertVariant.danger}
            isInline
            title="Please fix the errors below"
            style={{ marginBottom: '10px' }}
          />
        )}

        <Form className="pf-v5-c-form--compact">
          {/* Basic Fields */}
          {basicFields.map(renderField)}

          {/* Prompt Editor Button for Prompt and Eval nodes */}
          {needsPromptEditor && (
            <div style={{ 
              padding: '10px', 
              margin: '10px 0', 
              background: '#e7f1fa', 
              borderRadius: '4px',
              border: '1px solid #bee1f4',
            }}>
              <div style={{ marginBottom: '4px', fontWeight: 500, fontSize: '13px' }}>
                {config.prompt_config_path ? 'Prompt Configured' : 'Configure Prompt Template'}
              </div>
              <div style={{ fontSize: '12px', color: '#6a6e73', marginBottom: '8px' }}>
                {config.prompt_config_path 
                  ? `Current prompt: ${config.prompt_config_path}`
                  : 'Define the system and user message templates for this node.'
                }
              </div>
              <Button
                variant={config.prompt_config_path ? 'secondary' : 'primary'}
                icon={<EditIcon />}
                onClick={handleOpenPromptEditor}
                size="sm"
              >
                {config.prompt_config_path ? 'Edit Prompt' : 'Create Prompt'}
              </Button>
            </div>
          )}

          {/* Advanced Fields */}
          {advancedFields.length > 0 && (
            <>
              <Divider style={{ margin: '10px 0' }} />
              <ExpandableSection
                toggleText={showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options'}
                isExpanded={showAdvanced}
                onToggle={() => setShowAdvanced(!showAdvanced)}
              >
                <div style={{ paddingTop: '10px' }}>
                  {advancedFields.map(renderField)}
                </div>
              </ExpandableSection>
            </>
          )}
        </Form>
      </div>

      {/* Footer Actions - Compact */}
      <div style={{ 
        padding: '10px 12px', 
        borderTop: '1px solid #d2d2d2',
        background: '#f5f5f5',
        flexShrink: 0,
      }}>
        <Split hasGutter>
          <SplitItem>
            <Button
              variant="danger"
              icon={<TrashIcon />}
              onClick={onDelete}
              size="sm"
            >
              Delete
            </Button>
          </SplitItem>
          <SplitItem isFilled />
          <SplitItem>
            <Button variant="secondary" onClick={onClose} size="sm">
              Cancel
            </Button>
          </SplitItem>
          <SplitItem>
            <Button
              variant="primary"
              icon={<SaveIcon />}
              onClick={handleSave}
              size="sm"
              data-tour="save-button"
            >
              Save
            </Button>
          </SplitItem>
        </Split>
      </div>

      {/* Prompt Editor Modal */}
      {showPromptEditor && (
        <PromptEditorModal
          bundleType={node.type === NODE_TYPES.EVAL ? 'evaluation' : 'generation'}
          promptName={config.block_name || 'custom_prompt'}
          initialPrompt={config.system_message && config.user_message ? [
            { role: 'system', content: config.system_message },
            { role: 'user', content: config.user_message },
          ] : null}
          existingPromptPath={config.prompt_config_path ? 
            (existingFlowName || tempFlowName 
              ? `custom_flows/${(existingFlowName || tempFlowName).replace(' (Custom)', '').replace(' (Copy)', '').toLowerCase().replace(/ /g, '_').replace(/-/g, '_')}/${config.prompt_config_path}`
              : config.prompt_config_path
            ) : null
          }
          onSave={handlePromptSave}
          onClose={() => setShowPromptEditor(false)}
        />
      )}
    </div>
  );
};

export default NodeConfigDrawer;
