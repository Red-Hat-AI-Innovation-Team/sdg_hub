import React from 'react';
import { CutIcon } from '@patternfly/react-icons';

/**
 * Parser Node Configuration Component
 * Used in the NodeConfigDrawer for configuring parser nodes
 */
export const ParserNodeConfig = {
  type: 'parser',
  label: 'Parser',
  icon: CutIcon,
  color: '#f0ab00',
  description: 'Extract structured data from LLM responses using tags or patterns',
  
  // Default configuration values
  defaultConfig: {
    block_name: '',
    input_cols: '',
    output_cols: [],
    start_tags: [],
    end_tags: [],
    parsing_pattern: '',
    parser_cleanup_tags: [],
    save_reasoning_content: false,
  },

  // Configuration fields definition
  fields: [
    {
      name: 'block_name',
      label: 'Block Name',
      type: 'text',
      required: true,
      placeholder: 'e.g., parse_summary',
      helperText: 'A unique identifier for this parser block',
    },
    {
      name: 'input_cols',
      label: 'Input Column',
      type: 'text',
      required: true,
      placeholder: 'e.g., extract_gen_summary_content',
      helperText: 'Column containing the text to parse',
    },
    {
      name: 'output_cols',
      label: 'Output Columns',
      type: 'tags',
      required: true,
      placeholder: 'Add output column...',
      helperText: 'Names for extracted values (press Enter to add)',
    },
    {
      name: 'start_tags',
      label: 'Start Tags',
      type: 'tags',
      required: false,
      placeholder: 'Add start tag...',
      helperText: 'Tags marking the start of content to extract (e.g., [QUESTION])',
    },
    {
      name: 'end_tags',
      label: 'End Tags',
      type: 'tags',
      required: false,
      placeholder: 'Add end tag...',
      helperText: 'Tags marking the end of content to extract (e.g., [END])',
    },
    {
      name: 'parsing_pattern',
      label: 'Regex Pattern',
      type: 'textarea',
      required: false,
      rows: 2,
      placeholder: '\\[QUESTION\\]\\s*(.*?)\\s*\\[ANSWER\\]\\s*(.*?)\\s*(?=\\[QUESTION\\]|$)',
      helperText: 'Advanced: Use regex pattern instead of tags for complex extraction',
      advanced: true,
    },
    {
      name: 'parser_cleanup_tags',
      label: 'Cleanup Tags',
      type: 'tags',
      required: false,
      placeholder: 'Add cleanup tag...',
      helperText: 'Tags to remove from extracted content',
      advanced: true,
    },
    {
      name: 'save_reasoning_content',
      label: 'Save Reasoning Content',
      type: 'checkbox',
      defaultValue: false,
      helperText: 'Save any reasoning content from the model response',
      advanced: true,
    },
  ],

  // Validation function
  validate: (config) => {
    const errors = {};
    if (!config.block_name?.trim()) {
      errors.block_name = 'Block name is required';
    }
    if (!config.input_cols?.trim()) {
      errors.input_cols = 'Input column is required';
    }
    if (!config.output_cols || config.output_cols.length === 0) {
      errors.output_cols = 'At least one output column is required';
    }
    // Must have either tags or pattern
    const hasTags = config.start_tags?.length > 0 || config.end_tags?.length > 0;
    const hasPattern = config.parsing_pattern?.trim();
    if (!hasTags && !hasPattern) {
      errors.start_tags = 'Either tags or regex pattern is required';
    }
    return errors;
  },

  // Generate block configuration for serialization
  toBlockConfig: (config) => {
    // Helper to filter out only null/undefined values (keep empty strings - they're valid for "extract until end")
    const cleanTagsKeepEmpty = (tags) => {
      if (!Array.isArray(tags)) return [];
      return tags.filter(tag => tag !== null && tag !== undefined);
    };
    
    // Helper to filter out null/undefined AND empty strings (for cleanup tags where empty makes no sense)
    const cleanTags = (tags) => {
      if (!Array.isArray(tags)) return [];
      return tags.filter(tag => tag !== null && tag !== undefined && tag !== '');
    };
    
    // Determine block type: RegexParserBlock when using a regex pattern, TagParserBlock when using tags
    const hasPattern = config.parsing_pattern?.trim();

    const blockConfig = {
      block_type: hasPattern ? 'RegexParserBlock' : 'TagParserBlock',
      block_config: {
        block_name: config.block_name,
        input_cols: config.input_cols,
        output_cols: Array.isArray(config.output_cols)
          ? (config.output_cols.length === 1 ? config.output_cols[0] : config.output_cols)
          : config.output_cols,
      },
    };

    if (hasPattern) {
      blockConfig.block_config.parsing_pattern = config.parsing_pattern;
    } else {
      // Add tags if provided - keep empty strings as they mean "extract until end of text"
      // But filter out null/undefined values
      let startTags = cleanTagsKeepEmpty(config.start_tags);
      let endTags = cleanTagsKeepEmpty(config.end_tags);

      // Pydantic requires start_tags and end_tags to have the same length
      if (startTags.length > 0 && endTags.length < startTags.length) {
        while (endTags.length < startTags.length) {
          endTags.push('');
        }
      }
      if (endTags.length > 0 && startTags.length < endTags.length) {
        while (startTags.length < endTags.length) {
          startTags.push('');
        }
      }

      if (startTags.length > 0) {
        blockConfig.block_config.start_tags = startTags;
      }
      if (endTags.length > 0) {
        blockConfig.block_config.end_tags = endTags;
      }
    }

    // Add optional fields
    const cleanupTags = cleanTags(config.parser_cleanup_tags);
    if (cleanupTags.length > 0) {
      blockConfig.block_config.parser_cleanup_tags = cleanupTags;
    }
    if (config.save_reasoning_content) {
      blockConfig.block_config.save_reasoning_content = true;
    }

    return blockConfig;
  },
};

/**
 * Parser Node Preview Component
 * Displayed in the node on the canvas
 */
export const ParserNodePreview = ({ config }) => {
  const outputs = config.output_cols || [];
  const hasTags = (config.start_tags?.length > 0) || (config.end_tags?.length > 0);
  const hasPattern = config.parsing_pattern?.trim();

  return (
    <div style={{ padding: '4px 8px', fontSize: '11px' }}>
      <div style={{ color: '#6a6e73', marginBottom: '2px' }}>
        {hasTags 
          ? `${config.start_tags?.length || 0} tag pair(s)`
          : hasPattern 
            ? 'Regex pattern'
            : 'No extraction defined'
        }
      </div>
      <div style={{ color: '#151515', fontWeight: 500 }}>
        → {outputs.length > 0 
          ? outputs.slice(0, 2).join(', ') + (outputs.length > 2 ? '...' : '')
          : 'output'
        }
      </div>
    </div>
  );
};

export default ParserNodeConfig;
