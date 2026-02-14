import React from 'react';
import { CubesIcon } from '@patternfly/react-icons';

/**
 * LLM Node Configuration Component
 * 
 * This node combines:
 * - PromptBuilderBlock (build prompt from template)
 * - LLMChatBlock (send prompt to LLM)
 * - LLMResponseExtractorBlock (extract content from response)
 * 
 * This matches the actual flow.yaml pattern where these 3 blocks
 * are always used together.
 */
export const LLMNodeConfig = {
  type: 'llm',
  label: 'LLM',
  icon: CubesIcon,
  color: '#0066cc',
  description: 'Build prompt + call LLM + extract response',
  
  // Default configuration values
  defaultConfig: {
    block_name: '',
    input_cols: [],
    output_cols: '',
    system_message: 'You are a helpful AI assistant.',
    user_message: '',
    max_tokens: 2048,
    temperature: 0.7,
    n: 1,
    async_mode: true,
  },

  // Configuration fields definition
  fields: [
    {
      name: 'block_name',
      label: 'Block Name',
      type: 'text',
      required: true,
      placeholder: 'e.g., gen_summary',
      helperText: 'A unique identifier for this LLM block',
    },
    {
      name: 'input_cols',
      label: 'Input Variables',
      type: 'tags',
      required: true,
      placeholder: 'e.g., document, domain',
      helperText: 'Column names from your dataset. Each variable will be auto-added as {{variable_name}} to the User Message Template.',
      autoAddToUserMessage: true, // Flag to trigger auto-append behavior
    },
    {
      name: 'output_cols',
      label: 'Output Column',
      type: 'text',
      required: true,
      placeholder: 'e.g., summary',
      helperText: 'Column name for the extracted LLM response',
    },
    {
      name: 'system_message',
      label: 'System Message',
      type: 'textarea',
      required: false,
      rows: 2,
      placeholder: 'You are a helpful AI assistant...',
      helperText: 'Sets the AI behavior and persona for this task.',
      hasInfoTooltip: true,
      infoTooltipContent: `**What is a System Message?**

The system message defines the AI's role, personality, and behavior. It's like giving instructions to an assistant before they start working.

**What to include:**
• The AI's role or expertise (e.g., "You are an expert data analyst")
• How it should respond (tone, format, detail level)
• Any constraints or guidelines to follow

**Examples:**

\`\`\`
You are a knowledgeable AI assistant that helps users understand complex topics in simple terms.
\`\`\`

\`\`\`
You are an expert in {{domain}}. Provide detailed, accurate responses based on the given context. Always structure your output using the specified tags.
\`\`\`

\`\`\`
You are a helpful assistant specialized in text analysis. Be concise and focus on the most important insights.
\`\`\`

**Note:** The system message is optional but recommended for better quality responses.`,
    },
    {
      name: 'user_message',
      label: 'User Message Template',
      type: 'textarea',
      required: true,
      rows: 5,
      placeholder: 'Example:\nAnalyze the following text and provide a summary.\n\n{{document}}\n\nRespond with:\n[SUMMARY]\nYour summary here\n[/SUMMARY]',
      helperText: 'Input variables (e.g., {{document}}) are automatically added from your Input Variables above.',
      hasInfoTooltip: true,
      infoTooltipContent: `**Tip: Use tags for parsing**

If you want to extract specific parts of the LLM response later, include tags in your prompt template. Then connect a **Parser node** to extract the tagged content.

**Example prompt with tags:**
\`\`\`
Analyze the following document about {{domain}}:

{{document}}

Provide your analysis in this format:
[SUMMARY]
A brief summary of the main points
[/SUMMARY]

[KEYWORDS]
Key terms and concepts
[/KEYWORDS]
\`\`\`

**Notes:**
• Input variables like \`{{document}}\` are added automatically when you add Input Variables above
• Common tag patterns: \`[TAG]...[/TAG]\` or \`[Start of TAG]...[End of TAG]\`
• The Parser node will extract content between your specified start and end tags`,
    },
    {
      name: 'temperature',
      label: 'Temperature',
      type: 'slider',
      required: false,
      min: 0,
      max: 2,
      step: 0.1,
      defaultValue: 0.7,
      helperText: 'Controls randomness (0 = deterministic, 2 = most random)',
      advanced: true,
    },
    {
      name: 'max_tokens',
      label: 'Max Tokens',
      type: 'number',
      required: false,
      min: 1,
      max: 16384,
      step: 100,
      defaultValue: 2048,
      helperText: 'Maximum number of tokens in the response',
      advanced: true,
    },
    {
      name: 'n',
      label: 'Number of Responses',
      type: 'number',
      required: false,
      min: 1,
      max: 100,
      step: 1,
      defaultValue: 1,
      helperText: 'Number of responses to generate per prompt',
      advanced: true,
    },
    {
      name: 'async_mode',
      label: 'Async Mode',
      type: 'checkbox',
      defaultValue: true,
      helperText: 'Enable asynchronous processing (faster for large batches)',
      advanced: true,
    },
  ],

  // Validation function
  validate: (config) => {
    const errors = {};
    if (!config.block_name?.trim()) {
      errors.block_name = 'Block name is required';
    }
    if (!config.input_cols || config.input_cols.length === 0) {
      errors.input_cols = 'At least one input variable is required';
    }
    if (!config.output_cols?.trim()) {
      errors.output_cols = 'Output column is required';
    }
    if (!config.user_message?.trim()) {
      errors.user_message = 'User message template is required';
    }
    if (config.max_tokens < 1 || config.max_tokens > 16384) {
      errors.max_tokens = 'Max tokens must be between 1 and 16384';
    }
    if (config.temperature < 0 || config.temperature > 2) {
      errors.temperature = 'Temperature must be between 0 and 2';
    }
    return errors;
  },

  // Generate block configurations for serialization
  // LLM Node normally generates THREE blocks: PromptBuilderBlock + LLMChatBlock + LLMResponseExtractorBlock
  // When loaded from a template (_skipPromptBuilder=true), only generate LLMChat + Extractor
  // because the PromptBuilder is already a separate PROMPT node
  toBlockConfig: (config, promptConfigPath) => {
    // Check if we have preserved configs from a template
    // BUT: If the block_name has changed from the original, we need to regenerate column names
    const originalBlockName = config._llm_block_config?.block_name;
    const blockNameChanged = originalBlockName && originalBlockName !== config.block_name;
    
    const hasOriginalLLMConfig = !!config._llm_block_config && !blockNameChanged;
    const hasOriginalExtractorConfig = !!config._extractor_block_config && !blockNameChanged;
    const skipPromptBuilder = !!config._skipPromptBuilder;
    
    // Determine prompt file path:
    // - If node was renamed, use new block_name.yaml (the edited prompt was saved there)
    // - Otherwise, extract just the filename from the potentially absolute path
    let effectivePromptPath;
    if (blockNameChanged) {
      // Node was renamed - prompt was saved to new file matching the new name
      effectivePromptPath = `${config.block_name}.yaml`;
    } else if (promptConfigPath || config.prompt_config_path) {
      // Extract just the filename from potentially absolute path
      const fullPath = promptConfigPath || config.prompt_config_path;
      effectivePromptPath = fullPath.split('/').pop();
    } else {
      effectivePromptPath = null;
    }
    
    // Determine column names - use original if available AND not renamed, otherwise generate
    let promptColName, rawOutputColName;
    
    if (hasOriginalLLMConfig) {
      // Use the original LLM block's input/output columns
      promptColName = config._llm_block_config.input_cols;
      rawOutputColName = config._llm_block_config.output_cols;
    } else {
      // Generate new column names based on current block_name
      promptColName = `${config.block_name}_prompt`;
      rawOutputColName = `raw_${config.output_cols || config.block_name}_prompt`;
    }
    
    const blocks = [];
    
    // 1. PromptBuilderBlock - only if not loaded from template
    if (!skipPromptBuilder) {
      const promptBuilderConfig = {
        block_name: `${config.block_name}_prompt`,
        input_cols: config.input_cols,
        output_cols: promptColName,
      };
      
      if (effectivePromptPath) {
        promptBuilderConfig.prompt_config_path = effectivePromptPath;
      } else {
        promptBuilderConfig.prompt_config = {
          messages: [
            ...(config.system_message ? [{ role: 'system', content: config.system_message }] : []),
            { role: 'user', content: config.user_message },
          ],
        };
        promptBuilderConfig.prompt_config_path = `${config.block_name}_prompt.yaml`;
      }
      
      blocks.push({
        block_type: 'PromptBuilderBlock',
        block_config: promptBuilderConfig,
      });
    }
    
    // 2. LLMChatBlock - always use current block_name, but preserve other settings
    const llmBlockConfig = {
      block_name: config.block_name,
      input_cols: promptColName,
      output_cols: rawOutputColName,
      max_tokens: config.max_tokens ?? config._llm_block_config?.max_tokens ?? 2048,
      temperature: config.temperature ?? config._llm_block_config?.temperature ?? 0.7,
      n: config.n ?? config._llm_block_config?.n ?? 1,
      async_mode: config.async_mode ?? config._llm_block_config?.async_mode ?? true,
    };
    
    blocks.push({
      block_type: 'LLMChatBlock',
      block_config: llmBlockConfig,
    });
    
    // 3. LLMResponseExtractorBlock - use preserved config from original flow when
    //    available (e.g. field_prefix, block_name), otherwise derive from current block_name
    const origExtractor = config._extractor_block_config || {};
    const extractorBlockConfig = {
      block_name: origExtractor.block_name || `extract_${config.block_name}`,
      input_cols: rawOutputColName,
      output_cols: origExtractor.output_cols || `${config.output_cols || config.block_name}_content`,
      extract_content: origExtractor.extract_content ?? true,
      expand_lists: origExtractor.expand_lists ?? (config.n > 1),
    };
    // Preserve field_prefix if the original flow defined one - this controls
    // the output column naming (e.g. field_prefix: "topic_" → "topic_content")
    if (origExtractor.field_prefix) {
      extractorBlockConfig.field_prefix = origExtractor.field_prefix;
    }
    
    blocks.push({
      block_type: 'LLMResponseExtractorBlock',
      block_config: extractorBlockConfig,
    });
    
    return blocks;
  },
};

/**
 * LLM Node Preview Component
 * Displayed in the node on the canvas
 */
export const LLMNodePreview = ({ config }) => {
  const inputVars = Array.isArray(config.input_cols) 
    ? config.input_cols.join(', ') 
    : config.input_cols || '';
  
  return (
    <div style={{ padding: '4px 8px', fontSize: '11px' }}>
      <div style={{ color: '#6a6e73', marginBottom: '2px' }}>
        {inputVars ? `{{${inputVars.split(',')[0]?.trim()}}}...` : 'No inputs'}
      </div>
      <div style={{ color: '#151515', fontWeight: 500 }}>
        → {config.output_cols || 'output'}
      </div>
    </div>
  );
};

export default LLMNodeConfig;
