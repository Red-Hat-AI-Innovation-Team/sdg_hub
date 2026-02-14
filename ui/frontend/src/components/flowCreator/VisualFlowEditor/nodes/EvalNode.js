import React from 'react';
import { CheckCircleIcon } from '@patternfly/react-icons';

/**
 * Eval Node Configuration Component
 * Used in the NodeConfigDrawer for configuring evaluation nodes
 * 
 * Eval nodes are special - they auto-generate multiple blocks:
 * PromptBuilder + LLMChat + LLMResponseExtractor + TextParser + ColumnValueFilter
 */
export const EvalNodeConfig = {
  type: 'eval',
  label: 'Eval',
  icon: CheckCircleIcon,
  color: '#c9190b',
  description: 'Evaluate quality and filter results based on LLM judgment',
  
  // Default configuration values
  defaultConfig: {
    block_name: '',
    input_cols: [],
    system_message: 'You are an AI assistant specialized in evaluating content quality.',
    user_message: '',
    explanation_start_tag: '[Start of Explanation]',
    explanation_end_tag: '[End of Explanation]',
    judgment_start_tag: '[Start of Answer]',
    judgment_end_tag: '[End of Answer]',
    filter_value: 'YES',
    filter_operation: 'eq',
    filter_dtype: null, // null for string, 'float' or 'int' for numeric
    max_tokens: 2048,
    temperature: 0.0,
  },

  // Configuration fields definition
  fields: [
    {
      name: 'block_name',
      label: 'Evaluation Name',
      type: 'text',
      required: true,
      placeholder: 'e.g., eval_faithfulness',
      helperText: 'A unique identifier for this evaluation',
    },
    {
      name: 'input_cols',
      label: 'Input Variables',
      type: 'tags',
      required: true,
      placeholder: 'Add variable...',
      helperText: 'Variables to include in evaluation (e.g., document, response)',
      suggestions: ['document', 'response', 'question', 'context', 'answer'],
    },
    {
      name: 'system_message',
      label: 'System Message',
      type: 'textarea',
      required: true,
      rows: 2,
      placeholder: 'You are an AI assistant specialized in evaluating...',
      helperText: 'System role message for the evaluation prompt',
    },
    {
      name: 'user_message',
      label: 'Evaluation Prompt',
      type: 'textarea',
      required: true,
      rows: 5,
      placeholder: `Evaluate the following content for faithfulness to the source document.

Document:
{{document}}

Answer:
{{response}}

[Start of Explanation]
Explain your reasoning here...
[End of Explanation]

[Start of Answer]
YES or NO
[End of Answer]`,
      helperText: 'Use {{variable}} for template variables. Include tags for parsing.',
    },
    {
      name: 'explanation_start_tag',
      label: 'Explanation Start Tag',
      type: 'text',
      required: true,
      defaultValue: '[Start of Explanation]',
      helperText: 'Tag marking start of explanation',
    },
    {
      name: 'explanation_end_tag',
      label: 'Explanation End Tag',
      type: 'text',
      required: true,
      defaultValue: '[End of Explanation]',
      helperText: 'Tag marking end of explanation',
    },
    {
      name: 'judgment_start_tag',
      label: 'Judgment Start Tag',
      type: 'text',
      required: true,
      defaultValue: '[Start of Answer]',
      helperText: 'Tag marking start of judgment/score',
    },
    {
      name: 'judgment_end_tag',
      label: 'Judgment End Tag',
      type: 'text',
      required: true,
      defaultValue: '[End of Answer]',
      helperText: 'Tag marking end of judgment/score',
    },
    {
      name: 'filter_value',
      label: 'Passing Value',
      type: 'text',
      required: true,
      placeholder: 'e.g., YES or 2.0',
      helperText: 'Value that indicates passing (e.g., YES, 4, 2.0)',
    },
    {
      name: 'filter_operation',
      label: 'Filter Operation',
      type: 'select',
      required: true,
      options: [
        { value: 'eq', label: 'Equals (==)' },
        { value: 'ne', label: 'Not Equals (!=)' },
        { value: 'gt', label: 'Greater Than (>)' },
        { value: 'ge', label: 'Greater or Equal (>=)' },
        { value: 'lt', label: 'Less Than (<)' },
        { value: 'le', label: 'Less or Equal (<=)' },
      ],
      defaultValue: 'eq',
      helperText: 'How to compare against passing value',
    },
    {
      name: 'filter_dtype',
      label: 'Value Type',
      type: 'select',
      required: false,
      options: [
        { value: '', label: 'String (text)' },
        { value: 'float', label: 'Float (decimal number)' },
        { value: 'int', label: 'Integer (whole number)' },
      ],
      defaultValue: '',
      helperText: 'Type to convert judgment to before filtering',
      advanced: true,
    },
    {
      name: 'max_tokens',
      label: 'Max Tokens',
      type: 'number',
      min: 1,
      max: 4096,
      defaultValue: 2048,
      helperText: 'Maximum response length for evaluation',
      advanced: true,
    },
    {
      name: 'temperature',
      label: 'Temperature',
      type: 'number',
      min: 0,
      max: 2,
      step: 0.1,
      defaultValue: 0.0,
      helperText: 'Usually 0 for consistent evaluations',
      advanced: true,
    },
  ],

  // Validation function
  validate: (config) => {
    const errors = {};
    if (!config.block_name?.trim()) {
      errors.block_name = 'Evaluation name is required';
    }
    if (!config.input_cols || config.input_cols.length === 0) {
      errors.input_cols = 'At least one input variable is required';
    }
    if (!config.user_message?.trim()) {
      errors.user_message = 'Evaluation prompt is required';
    }
    if (!config.filter_value?.toString().trim()) {
      errors.filter_value = 'Passing value is required';
    }
    return errors;
  },

  // Generate block configurations for serialization
  // Eval Node generates MULTIPLE blocks
  toBlockConfig: (config, promptFileName) => {
    const baseName = config.block_name;
    
    // Use prompt_config_path from config if available (from template), 
    // then promptFileName arg, then fallback to default
    const effectivePromptPath = config.prompt_config_path || promptFileName || `${baseName}.yaml`;
    
    // Handle both formats for tags:
    // 1. Individual fields: explanation_start_tag, judgment_start_tag, etc. (from user input)
    // 2. Arrays: start_tags, end_tags (from loaded templates)
    const getStartTags = () => {
      // Check for array format first (from templates)
      if (Array.isArray(config.start_tags) && config.start_tags.length >= 2) {
        // Filter out null/undefined values and use defaults if needed
        return [
          config.start_tags[0] || '[Start of Explanation]',
          config.start_tags[1] || '[Start of Answer]'
        ];
      }
      // Fall back to individual fields
      return [
        config.explanation_start_tag || '[Start of Explanation]',
        config.judgment_start_tag || '[Start of Answer]'
      ];
    };
    
    const getEndTags = () => {
      // Check for array format first (from templates)
      if (Array.isArray(config.end_tags) && config.end_tags.length >= 2) {
        // Filter out null/undefined values and use defaults if needed
        return [
          config.end_tags[0] || '[End of Explanation]',
          config.end_tags[1] || '[End of Answer]'
        ];
      }
      // Fall back to individual fields
      return [
        config.explanation_end_tag || '[End of Explanation]',
        config.judgment_end_tag || '[End of Answer]'
      ];
    };
    
    return [
      // 1. Prompt Builder
      {
        block_type: 'PromptBuilderBlock',
        block_config: {
          block_name: `${baseName}_prompt`,
          input_cols: config.input_cols,
          output_cols: `${baseName}_prompt`,
          prompt_config_path: effectivePromptPath,
          format_as_messages: true,
        },
      },
      // 2. LLM Chat
      {
        block_type: 'LLMChatBlock',
        block_config: {
          block_name: `${baseName}_llm`,
          input_cols: `${baseName}_prompt`,
          output_cols: `${baseName}_response`,
          max_tokens: config.max_tokens || 2048,
          temperature: config.temperature || 0.0,
          n: 1,
          async_mode: true,
        },
      },
      // 3. LLM Response Extractor
      {
        block_type: 'LLMResponseExtractorBlock',
        block_config: {
          block_name: `extract_${baseName}`,
          input_cols: `${baseName}_response`,
          extract_content: true,
        },
      },
      // 4. Text Parser
      {
        block_type: 'TextParserBlock',
        block_config: {
          block_name: `parse_${baseName}`,
          input_cols: `extract_${baseName}_content`,
          output_cols: [`${baseName}_explanation`, `${baseName}_judgment`],
          start_tags: getStartTags(),
          end_tags: getEndTags(),
        },
      },
      // 5. Column Value Filter
      {
        block_type: 'ColumnValueFilterBlock',
        block_config: {
          block_name: `${baseName}_filter`,
          input_cols: [`${baseName}_judgment`],
          filter_value: config.filter_value || 'YES',
          operation: config.filter_operation || 'eq',
          ...(config.filter_dtype && { convert_dtype: config.filter_dtype }),
        },
      },
    ];
  },
};

/**
 * Eval Node Preview Component
 * Displayed in the node on the canvas
 */
export const EvalNodePreview = ({ config }) => {
  const filterOps = {
    eq: '==',
    ne: '!=',
    gt: '>',
    ge: '>=',
    lt: '<',
    le: '<=',
  };
  const op = filterOps[config.filter_operation] || '==';

  return (
    <div style={{ padding: '4px 8px', fontSize: '11px' }}>
      <div style={{ color: '#6a6e73', marginBottom: '2px' }}>
        Pass: judgment {op} {config.filter_value || 'YES'}
      </div>
      <div style={{ color: '#c9190b', fontWeight: 500 }}>
        Auto-filters results
      </div>
    </div>
  );
};

export default EvalNodeConfig;
