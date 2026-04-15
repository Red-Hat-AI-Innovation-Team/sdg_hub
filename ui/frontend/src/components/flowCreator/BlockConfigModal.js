import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Form,
  FormGroup,
  TextInput,
  NumberInput,
  Checkbox,
  Select,
  SelectOption,
  SelectList,
  MenuToggle,
  Alert,
  AlertVariant,
  ExpandableSection,
  Grid,
  GridItem,
  CodeBlock,
  CodeBlockCode,
  Accordion,
  AccordionItem,
  AccordionToggle,
  AccordionContent,
  Title,
  Badge,
} from '@patternfly/react-core';
import PromptEditorModal from './PromptEditorModal';
import { promptAPI } from '../../services/api';

/**
 * Block Configuration Modal
 * 
 * Shows ALL configuration parameters for a block or bundle
 * Pre-populated with default values
 * User configures BEFORE adding to flow
 */
const BlockConfigModal = ({ block, isEdit, onSubmit, onClose, onTempFlowCreated, existingFlowName }) => {
  const [config, setConfig] = useState({});
  const [bundleBlocks, setBundleBlocks] = useState([]);
  const [expandedBlocks, setExpandedBlocks] = useState([]);
  const [isAdvancedExpanded, setIsAdvancedExpanded] = useState(false);
  const [selectOpen, setSelectOpen] = useState({});
  
  // Multi-step state for bundles with prompts
  const [currentStep, setCurrentStep] = useState('config'); // 'prompts' or 'config'
  const [promptsToEdit, setPromptsToEdit] = useState([]);
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
  const [savedPrompts, setSavedPrompts] = useState({});

  /**
   * Initialize config with defaults
   */
  useEffect(() => {
    if (block) {
      if (block.isBundle) {
        // Generate the blocks this bundle will create with full default configs
        const blocks = generateBundleBlocksWithDefaults(block);
        setBundleBlocks(blocks);
        
        // Check if any blocks need prompt configuration (all PromptBuilderBlocks need prompts)
        const promptBlocks = blocks.filter(b => 
          b.block_type === 'PromptBuilderBlock'
        );
        
        if (promptBlocks.length > 0 && !isEdit) {
          // Set up prompt editing flow (only for new bundles, not editing)
          setPromptsToEdit(promptBlocks.map((b, index) => ({
            blockIndex: blocks.indexOf(b),
            promptName: b.block_config.prompt_config_path ? b.block_config.prompt_config_path.replace('.yaml', '') : '',
            blockName: b.block_config.block_name,
          })));
          setCurrentStep('prompts');
          setCurrentPromptIndex(0);
        } else {
          // No prompts needed or editing existing, go straight to config
          setCurrentStep('config');
          setExpandedBlocks([0]);
        }
      } else {
        // Initialize individual block with default config
        // Priority: block.block_config (for edited blocks) > block.defaultConfig?.block_config (for new blocks from library)
        const blockConfig = block.block_config || block.defaultConfig?.block_config || {};
        setConfig(blockConfig);
        
        // Check if this is a PromptBuilder block being edited
        if (isEdit && (block.block_type === 'PromptBuilderBlock' || block.defaultConfig?.block_type === 'PromptBuilderBlock')) {
          const promptPath = blockConfig.prompt_config_path;
          if (promptPath) {
            // Load the existing prompt and show editor first
            setPromptsToEdit([{
              blockIndex: 0,
              promptName: promptPath.replace('.yaml', ''),
              blockName: blockConfig.block_name,
              existingPrompt: true // Flag to load from file
            }]);
            setCurrentStep('prompts');
            setCurrentPromptIndex(0);
          } else {
            setCurrentStep('config');
          }
        } else {
          setCurrentStep('config');
        }
      }
    }
  }, [block, isEdit]);

  /**
   * Generate bundle blocks with full default configurations
   */
  const generateBundleBlocksWithDefaults = (bundle) => {
    const baseName = bundle.id;
    let blocks = [];

    switch (bundle.id) {
      case 'summary_generation':
        blocks = [
          {
            block_type: 'DuplicateColumnsBlock',
            block_config: {
              block_name: 'duplicate_document_col',
              input_cols: { document: 'base_document' },
            },
            displayName: 'Duplicate Columns',
            description: 'Preserves original document'
          },
          {
            block_type: 'PromptBuilderBlock',
            block_config: {
              block_name: `${baseName}_prompt`,
              input_cols: ['document'],
              output_cols: `${baseName}_prompt`,
              prompt_config_path: '', // Empty initially - user must define in prompt editor
            },
            displayName: 'Prompt Builder',
            description: 'Builds the summary prompt from input'
          },
          {
            block_type: 'LLMChatBlock',
            block_config: {
              block_name: `gen_${baseName}`,
              input_cols: `${baseName}_prompt`,
              output_cols: `raw_${baseName}`,
              max_tokens: 2048,
              temperature: 0.7,
              async_mode: true,
              n: 1,
            },
            displayName: 'LLM Chat',
            description: 'Generates summary using LLM'
          },
          {
            block_type: 'LLMParserBlock',
            block_config: {
              block_name: `${baseName}_parser`,
              input_cols: `raw_${baseName}`,
              extract_content: true,
            },
            displayName: 'LLM Parser',
            description: 'Parses LLM response'
          },
          {
            block_type: 'TagParserBlock',
            block_config: {
              block_name: `parse_${baseName}`,
              input_cols: `${baseName}_parser_content`,
              output_cols: 'summary',
              start_tags: [''],
              end_tags: [''],
            },
            displayName: 'Tag Parser',
            description: 'Extracts final summary text'
          }
        ];
        break;

      case 'generation':
        blocks = [
          {
            block_type: 'PromptBuilderBlock',
            block_config: {
              block_name: `${baseName}_prompt`,
              input_cols: ['document', 'domain'],
              output_cols: `${baseName}_prompt`,
              prompt_config_path: '', // Empty initially - user must define in prompt editor
            },
            displayName: 'Prompt Builder',
            description: 'Builds Q&A generation prompt'
          },
          {
            block_type: 'LLMChatBlock',
            block_config: {
              block_name: `gen_${baseName}`,
              input_cols: `${baseName}_prompt`,
              output_cols: `raw_${baseName}`,
              max_tokens: 2048,
              temperature: 1.0,
              async_mode: true,
              n: 1,
            },
            displayName: 'LLM Chat',
            description: 'Generates Q&A pairs using LLM'
          },
          {
            block_type: 'LLMParserBlock',
            block_config: {
              block_name: `${baseName}_parser`,
              input_cols: `raw_${baseName}`,
              extract_content: true,
            },
            displayName: 'LLM Parser',
            description: 'Parses LLM response'
          },
          {
            block_type: 'RegexParserBlock',
            block_config: {
              block_name: `parse_${baseName}`,
              input_cols: `${baseName}_parser_content`,
              output_cols: ['question', 'response'],
              parsing_pattern: '\\[(?:Question|QUESTION)\\]\\s*(.*?)\\s*\\[(?:Answer|ANSWER)\\]\\s*(.*?)\\s*(?=\\[(?:Question|QUESTION)\\]|$)',
            },
            displayName: 'Regex Parser',
            description: 'Extracts questions and answers'
          }
        ];
        break;

      case 'evaluation':
        blocks = [
          {
            block_type: 'PromptBuilderBlock',
            block_config: {
              block_name: `${baseName}_prompt`,
              input_cols: ['document', 'response'],
              output_cols: `${baseName}_prompt`,
              prompt_config_path: '', // Empty initially - user must define in prompt editor
              format_as_messages: true,
            },
            displayName: 'Prompt Builder',
            description: 'Builds evaluation prompt'
          },
          {
            block_type: 'LLMChatBlock',
            block_config: {
              block_name: `${baseName}_llm`,
              input_cols: `${baseName}_prompt`,
              output_cols: `${baseName}_response`,
              max_tokens: 2048,
              async_mode: true,
              n: 1,
            },
            displayName: 'LLM Chat',
            description: 'Evaluates content using LLM'
          },
          {
            block_type: 'LLMParserBlock',
            block_config: {
              block_name: `${baseName}_parser`,
              input_cols: `${baseName}_response`,
              extract_content: true,
            },
            displayName: 'LLM Parser',
            description: 'Parses evaluation response'
          },
          {
            block_type: 'TagParserBlock',
            block_config: {
              block_name: `parse_${baseName}`,
              input_cols: `${baseName}_parser_content`,
              output_cols: ['evaluation_explanation', 'evaluation_judgment'],
              start_tags: ['[Start of Explanation]', '[Start of Answer]'],
              end_tags: ['[End of Explanation]', '[End of Answer]'],
            },
            displayName: 'Tag Parser',
            description: 'Extracts explanation and judgment'
          },
          {
            block_type: 'ColumnValueFilterBlock',
            block_config: {
              block_name: `${baseName}_filter`,
              input_cols: ['evaluation_judgment'],
              filter_value: 'YES',
              operation: 'eq',
            },
            displayName: 'Filter',
            description: 'Filters by evaluation judgment'
          }
        ];
        break;

      default:
        blocks = [];
    }

    return blocks;
  };

  /**
   * Handle form field change
   */
  const handleChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  /**
   * Handle bundle block config change
   */
  const handleBundleBlockChange = (blockIndex, field, value) => {
    if (block.isBundle) {
      // For bundles, update the specific block in bundleBlocks array
      setBundleBlocks(prev => prev.map((b, i) => {
        if (i === blockIndex) {
          return {
            ...b,
            block_config: {
              ...b.block_config,
              [field]: value
            }
          };
        }
        return b;
      }));
    } else {
      // For single blocks, update the main config
      setConfig(prev => {
        const newConfig = { ...prev, [field]: value };
        return newConfig;
      });
    }
  };

  /**
   * Toggle block expansion in bundle config
   */
  const toggleBlockExpansion = (blockIndex) => {
    setExpandedBlocks(prev =>
      prev.includes(blockIndex)
        ? prev.filter(i => i !== blockIndex)
        : [...prev, blockIndex]
    );
  };

  /**
   * Handle prompt save
   */
  const handlePromptSave = async (promptData, promptName) => {
    try {
      // Use existing flow name if editing, otherwise generate/reuse temp name
      // Remove "(Custom)" suffix if present for the directory name
      const flowNameToUse = existingFlowName 
        ? existingFlowName.replace(' (Custom)', '').replace(' (Copy)', '')
        : (savedPrompts._tempFlowName || `temp_flow_${Date.now()}`);
      
      
      // Save prompt to server
      const response = await promptAPI.savePrompt({
        prompt_name: promptName,
        prompt_content: promptData,
        flow_name: flowNameToUse
      });
      
      
      // Only notify parent about temp flow if this is a new flow (not editing existing)
      if (!existingFlowName && onTempFlowCreated) {
        onTempFlowCreated(flowNameToUse);
      }
      
      // Store the saved prompt filename and flow name
      setSavedPrompts(prev => ({
        ...prev,
        [promptName]: response.prompt_filename,
        _tempFlowName: flowNameToUse // Store flow name for reuse
      }));
      
      // Update the corresponding block's prompt_config_path
      const promptInfo = promptsToEdit[currentPromptIndex];
      
      if (block.isBundle) {
        // For bundles, update bundleBlocks
        setBundleBlocks(prev => prev.map((b, i) => {
          if (i === promptInfo.blockIndex) {
            return {
              ...b,
              block_config: {
                ...b.block_config,
                prompt_config_path: response.prompt_filename
              }
            };
          }
          return b;
        }));
      } else {
        // For single block editing, update the config directly
        setConfig(prev => ({
          ...prev,
          prompt_config_path: response.prompt_filename
        }));
      }
      
      // Move to next prompt or to config step
      if (currentPromptIndex < promptsToEdit.length - 1) {
        setCurrentPromptIndex(currentPromptIndex + 1);
      } else {
        // All prompts done, move to config step
        setCurrentStep('config');
        setExpandedBlocks([0]);
      }
    } catch (error) {
      const msg = (error && error.message) || String(error) || 'Unknown error';
      console.error('Failed to save prompt:', msg);
      alert('Failed to save prompt: ' + msg);
    }
  };

  /**
   * Handle submit
   */
  const handleSubmit = () => {
    
    if (block.isBundle) {
      // Submit the configured bundle blocks (already expanded with user's configs)
      // Filter out any undefined/invalid blocks
      const blocksToSubmit = bundleBlocks
        .filter(b => b && b.block_type)
        .map(b => ({
          block_type: b.block_type,
          block_config: b.block_config || {}
        }));
      
      if (blocksToSubmit.length === 0) {
        alert('Error: No valid blocks configured. Please check your block configuration.');
        return;
      }
      
      onSubmit(blocksToSubmit);
    } else {
      // Submit single configured block
      const blockType = block.defaultConfig?.block_type || block.block_type;
      
      if (!blockType) {
        alert('Error: Block type is missing. Please try adding the block again.');
        return;
      }
      
      const configuredBlock = {
        block_type: blockType,
        block_config: config
      };
      onSubmit(configuredBlock);
    }
  };

  /**
   * Render configuration fields for a bundle block
   */
  const renderBundleBlockConfig = (bundleBlock, blockIndex) => {
    const blockConfig = bundleBlock.block_config;
    
    return (
      <>
        {/* Block Name */}
        <FormGroup label="Block Name" fieldId={`block-${blockIndex}-name`}>
          <TextInput
            type="text"
            id={`block-${blockIndex}-name`}
            value={blockConfig.block_name || ''}
            onChange={(event, value) => handleBundleBlockChange(blockIndex, 'block_name', value)}
          />
        </FormGroup>

        {/* Input Columns */}
        <FormGroup label="Input Columns" fieldId={`block-${blockIndex}-input`}>
          <TextInput
            type="text"
            id={`block-${blockIndex}-input`}
            value={
              bundleBlock.block_type === 'DuplicateColumnsBlock' && typeof blockConfig.input_cols === 'object' && !Array.isArray(blockConfig.input_cols)
                ? Object.entries(blockConfig.input_cols).map(([key, val]) => `${key}: ${val}`).join(', ')
                : Array.isArray(blockConfig.input_cols) 
                  ? blockConfig.input_cols.join(', ') 
                  : blockConfig.input_cols || ''
            }
            onChange={(event, value) => {
              // Special handling for DuplicateColumnsBlock - parse as dictionary
              if (bundleBlock.block_type === 'DuplicateColumnsBlock') {
                // Parse format: "source: dest" or "source1: dest1, source2: dest2"
                const pairs = value.split(',').map(s => s.trim()).filter(s => s);
                const dict = {};
                pairs.forEach(pair => {
                  const [key, val] = pair.split(':').map(s => s.trim());
                  if (key && val) {
                    dict[key] = val;
                  }
                });
                handleBundleBlockChange(blockIndex, 'input_cols', dict);
              } else {
                // Regular handling for other blocks
                const parsed = value.includes(',') ? value.split(',').map(s => s.trim()) : value;
                handleBundleBlockChange(blockIndex, 'input_cols', parsed);
              }
            }}
            helperText={
              bundleBlock.block_type === 'DuplicateColumnsBlock'
                ? 'Format: source_column: new_column_name (e.g., "document: base_document")'
                : undefined
            }
          />
        </FormGroup>

        {/* Output Columns */}
        <FormGroup label="Output Columns" fieldId={`block-${blockIndex}-output`}>
          <TextInput
            type="text"
            id={`block-${blockIndex}-output`}
            value={Array.isArray(blockConfig.output_cols) ? blockConfig.output_cols.join(', ') : blockConfig.output_cols || ''}
            onChange={(event, value) => {
              const parsed = value.includes(',') ? value.split(',').map(s => s.trim()) : value;
              handleBundleBlockChange(blockIndex, 'output_cols', parsed);
            }}
          />
        </FormGroup>

        {/* LLMChatBlock specific */}
        {bundleBlock.block_type === 'LLMChatBlock' && (
          <>
            <Grid hasGutter>
              <GridItem span={6}>
                <FormGroup label="Max Tokens" fieldId={`block-${blockIndex}-max-tokens`}>
                  <NumberInput
                    id={`block-${blockIndex}-max-tokens`}
                    value={blockConfig.max_tokens || 2048}
                    onMinus={() => handleBundleBlockChange(blockIndex, 'max_tokens', Math.max(1, (blockConfig.max_tokens || 2048) - 100))}
                    onPlus={() => handleBundleBlockChange(blockIndex, 'max_tokens', (blockConfig.max_tokens || 2048) + 100)}
                    onChange={(event) => {
                      const value = parseInt(event.target.value, 10);
                      handleBundleBlockChange(blockIndex, 'max_tokens', isNaN(value) ? 2048 : value);
                    }}
                    min={1}
                    widthChars={10}
                  />
                </FormGroup>
              </GridItem>
              <GridItem span={6}>
                <FormGroup label="Temperature" fieldId={`block-${blockIndex}-temperature`}>
                  <NumberInput
                    id={`block-${blockIndex}-temperature`}
                    value={blockConfig.temperature !== undefined ? blockConfig.temperature : 0.7}
                    onMinus={() => handleBundleBlockChange(blockIndex, 'temperature', Math.max(0, (blockConfig.temperature || 0.7) - 0.1))}
                    onPlus={() => handleBundleBlockChange(blockIndex, 'temperature', Math.min(2, (blockConfig.temperature || 0.7) + 0.1))}
                    onChange={(event) => {
                      const value = parseFloat(event.target.value);
                      handleBundleBlockChange(blockIndex, 'temperature', isNaN(value) ? 0.7 : value);
                    }}
                    min={0}
                    max={2}
                    widthChars={8}
                  />
                </FormGroup>
              </GridItem>
            </Grid>
            <Grid hasGutter>
              <GridItem span={6}>
                <FormGroup fieldId={`block-${blockIndex}-async`}>
                  <Checkbox
                    id={`block-${blockIndex}-async`}
                    label="Async Mode (faster for large batches)"
                    isChecked={blockConfig.async_mode !== undefined ? blockConfig.async_mode : true}
                    onChange={(event, checked) => handleBundleBlockChange(blockIndex, 'async_mode', checked)}
                  />
                </FormGroup>
              </GridItem>
              <GridItem span={6}>
                <FormGroup label="N (responses per request)" fieldId={`block-${blockIndex}-n`}>
                  <NumberInput
                    id={`block-${blockIndex}-n`}
                    value={blockConfig.n || 1}
                    onMinus={() => handleBundleBlockChange(blockIndex, 'n', Math.max(1, (blockConfig.n || 1) - 1))}
                    onPlus={() => handleBundleBlockChange(blockIndex, 'n', (blockConfig.n || 1) + 1)}
                    onChange={(event) => {
                      const value = parseInt(event.target.value, 10);
                      handleBundleBlockChange(blockIndex, 'n', isNaN(value) ? 1 : value);
                    }}
                    min={1}
                    widthChars={6}
                  />
                </FormGroup>
              </GridItem>
            </Grid>
          </>
        )}

        {/* TagParserBlock specific */}
        {bundleBlock.block_type === 'TagParserBlock' && (
          <Grid hasGutter>
            <GridItem span={6}>
              <FormGroup label="Start Tags (comma-separated)" fieldId={`block-${blockIndex}-start-tags`}>
                <TextInput
                  type="text"
                  id={`block-${blockIndex}-start-tags`}
                  value={Array.isArray(blockConfig.start_tags) ? blockConfig.start_tags.join(', ') : ''}
                  onChange={(event, value) => {
                    const parsed = value.split(',').map(s => s.trim());
                    handleBundleBlockChange(blockIndex, 'start_tags', parsed);
                  }}
                />
              </FormGroup>
            </GridItem>
            <GridItem span={6}>
              <FormGroup label="End Tags (comma-separated)" fieldId={`block-${blockIndex}-end-tags`}>
                <TextInput
                  type="text"
                  id={`block-${blockIndex}-end-tags`}
                  value={Array.isArray(blockConfig.end_tags) ? blockConfig.end_tags.join(', ') : ''}
                  onChange={(event, value) => {
                    const parsed = value.split(',').map(s => s.trim());
                    handleBundleBlockChange(blockIndex, 'end_tags', parsed);
                  }}
                />
              </FormGroup>
            </GridItem>
          </Grid>
        )}

        {/* RegexParserBlock specific */}
        {bundleBlock.block_type === 'RegexParserBlock' && (
          <FormGroup label="Parsing Pattern (regex)" fieldId={`block-${blockIndex}-parsing-pattern`}>
            <TextInput
              type="text"
              id={`block-${blockIndex}-parsing-pattern`}
              value={blockConfig.parsing_pattern || ''}
              onChange={(event, value) => handleBundleBlockChange(blockIndex, 'parsing_pattern', value)}
            />
          </FormGroup>
        )}

        {/* PromptBuilderBlock specific */}
        {bundleBlock.block_type === 'PromptBuilderBlock' && blockConfig.format_as_messages !== undefined && (
          <FormGroup fieldId={`block-${blockIndex}-format-messages`}>
            <Checkbox
              id={`block-${blockIndex}-format-messages`}
              label="Format as Messages"
              isChecked={blockConfig.format_as_messages}
              onChange={(event, checked) => handleBundleBlockChange(blockIndex, 'format_as_messages', checked)}
            />
          </FormGroup>
        )}

        {/* LLMParserBlock specific */}
        {bundleBlock.block_type === 'LLMParserBlock' && (
          <FormGroup fieldId={`block-${blockIndex}-extract-content`}>
            <Checkbox
              id={`block-${blockIndex}-extract-content`}
              label="Extract Content"
              isChecked={blockConfig.extract_content !== undefined ? blockConfig.extract_content : true}
              onChange={(event, checked) => handleBundleBlockChange(blockIndex, 'extract_content', checked)}
            />
          </FormGroup>
        )}

        {/* ColumnValueFilterBlock specific */}
        {bundleBlock.block_type === 'ColumnValueFilterBlock' && (
          <Grid hasGutter>
            <GridItem span={6}>
              <FormGroup label="Filter Value" fieldId={`block-${blockIndex}-filter-value`}>
                <TextInput
                  type="text"
                  id={`block-${blockIndex}-filter-value`}
                  value={blockConfig.filter_value !== undefined ? String(blockConfig.filter_value) : ''}
                  onChange={(event, value) => handleBundleBlockChange(blockIndex, 'filter_value', value)}
                />
              </FormGroup>
            </GridItem>
            <GridItem span={6}>
              <FormGroup label="Operation" fieldId={`block-${blockIndex}-operation`}>
                <TextInput
                  type="text"
                  id={`block-${blockIndex}-operation`}
                  value={blockConfig.operation || 'eq'}
                  onChange={(event, value) => handleBundleBlockChange(blockIndex, 'operation', value)}
                  placeholder="eq, ne, gt, lt, ge, le"
                />
              </FormGroup>
            </GridItem>
          </Grid>
        )}
      </>
    );
  };

  /**
   * Render form fields based on parameter type
   */
  const renderFormField = (key, param) => {
    switch (param.type) {
      case 'text':
        return (
          <TextInput
            type="text"
            id={key}
            value={config[key] || param.default || ''}
            onChange={(event, value) => handleChange(key, value)}
          />
        );
      
      case 'number':
        return (
          <NumberInput
            id={key}
            value={config[key] || param.default || 0}
            onMinus={() => handleChange(key, Math.max(param.min || 0, (config[key] || param.default) - (param.step || 1)))}
            onPlus={() => handleChange(key, Math.min(param.max || 10000, (config[key] || param.default) + (param.step || 1)))}
            onChange={(event) => {
              const value = parseFloat(event.target.value);
              handleChange(key, isNaN(value) ? param.default : value);
            }}
            min={param.min}
            max={param.max}
            widthChars={10}
          />
        );
      
      case 'boolean':
        return (
          <Checkbox
            id={key}
            label={param.label}
            isChecked={config[key] !== undefined ? config[key] : param.default}
            onChange={(event, checked) => handleChange(key, checked)}
          />
        );
      
      case 'select':
        return (
          <Select
            id={key}
            isOpen={selectOpen[key] || false}
            selected={config[key] || param.default}
            onSelect={(event, selection) => {
              handleChange(key, selection);
              setSelectOpen(prev => ({ ...prev, [key]: false }));
            }}
            onOpenChange={(isOpen) => setSelectOpen(prev => ({ ...prev, [key]: isOpen }))}
            toggle={(toggleRef) => (
              <MenuToggle ref={toggleRef} onClick={() => setSelectOpen(prev => ({ ...prev, [key]: !prev[key] }))}>
                {config[key] || param.default}
              </MenuToggle>
            )}
          >
            <SelectList>
              {param.options.map(option => (
                <SelectOption key={option} value={option}>
                  {option}
                </SelectOption>
              ))}
            </SelectList>
          </Select>
        );
      
      default:
        return null;
    }
  };

  if (!block) return null;

  // Show prompt editor if we're in the prompts step
  if (currentStep === 'prompts' && promptsToEdit.length > 0) {
    const currentPrompt = promptsToEdit[currentPromptIndex];
    const bundleTypeValue = block.id || block.defaultConfig?.block_type || block.block_type || 'custom';
    
    // Build the full prompt path including the flow directory
    let promptPath = null;
    if (currentPrompt.existingPrompt && config.prompt_config_path) {
      // If we have an existing flow name, construct the full path
      if (existingFlowName) {
        const flowDirName = existingFlowName.replace(' (Custom)', '').replace(' (Copy)', '').toLowerCase().replace(/ /g, '_').replace(/-/g, '_');
        promptPath = `custom_flows/${flowDirName}/${config.prompt_config_path}`;
      } else {
        promptPath = config.prompt_config_path;
      }
    }
    
    return (
      <PromptEditorModal
        bundleType={bundleTypeValue}
        promptName={currentPrompt.promptName}
        initialPrompt={null} // Will use example from PromptEditorModal
        existingPromptPath={promptPath}
        onSave={handlePromptSave}
        onClose={onClose}
      />
    );
  }

  // Show block configuration
  return (
    <Modal
      variant={ModalVariant.large}
      title={block.isBundle ? `Configure ${block.name} Bundle` : `Configure ${block.name} Block`}
      isOpen={true}
      onClose={onClose}
      actions={[
        <Button key="submit" variant="primary" onClick={handleSubmit}>
          {isEdit ? 'Update' : 'Add to Flow'}
        </Button>,
        <Button key="cancel" variant="link" onClick={onClose}>
          Cancel
        </Button>,
      ]}
    >
      {block.isBundle && (
        <Alert
          variant={AlertVariant.info}
          isInline
          title="Block Bundle"
          style={{ marginBottom: '1rem' }}
        >
          This bundle will create {block.blockCount} blocks in your flow: {block.generates.join(' → ')}
        </Alert>
      )}

      <Form>
        {/* Render bundle blocks in accordion OR single block config */}
        {block.isBundle ? (
          <Accordion asDefinitionList>
            {bundleBlocks.map((bundleBlock, index) => (
              <AccordionItem key={index}>
                <AccordionToggle
                  onClick={() => toggleBlockExpansion(index)}
                  isExpanded={expandedBlocks.includes(index)}
                  id={`block-${index}-toggle`}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Badge isRead>{index + 1}</Badge>
                    <strong>{bundleBlock.displayName}</strong>
                    <span style={{ color: '#6a6e73', fontSize: '0.875rem' }}>
                      ({bundleBlock.block_type})
                    </span>
                  </div>
                </AccordionToggle>
                <AccordionContent
                  id={`block-${index}-content`}
                  isHidden={!expandedBlocks.includes(index)}
                >
                  <div style={{ padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
                    <p style={{ marginBottom: '1rem', color: '#6a6e73' }}>
                      {bundleBlock.description}
                    </p>
                    {renderBundleBlockConfig(bundleBlock, index)}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        ) : (
          <>
            {/* Single block configuration - use same detailed config as bundles */}
            {(() => {
              const blockType = block.block_type || block.defaultConfig?.block_type;
              return renderBundleBlockConfig({ 
                block_type: blockType, 
                block_config: config,
                displayName: block.name || (blockType || '').replace('Block', ''),
                description: block.description || ''
              }, 0);
            })()}
          </>
        )}

        {/* Preview Configuration */}
        <ExpandableSection
          toggleText="Preview Configuration (YAML)"
          isExpanded={isAdvancedExpanded}
          onToggle={() => setIsAdvancedExpanded(!isAdvancedExpanded)}
        >
          <CodeBlock>
            <CodeBlockCode>
              {block.isBundle 
                ? JSON.stringify(bundleBlocks.map(b => ({ block_type: b.block_type, block_config: b.block_config })), null, 2)
                : JSON.stringify({ block_type: block.defaultConfig?.block_type, block_config: config }, null, 2)
              }
            </CodeBlockCode>
          </CodeBlock>
        </ExpandableSection>
      </Form>
    </Modal>
  );
};

export default BlockConfigModal;

