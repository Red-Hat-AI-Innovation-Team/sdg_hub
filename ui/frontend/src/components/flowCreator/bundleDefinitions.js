/**
 * Block Bundle Definitions
 * 
 * Shared bundle configurations used across the Flow Creator
 */

export const BLOCK_BUNDLES = [
  {
    id: 'summary_generation',
    name: 'Summarization',
    icon: '📝',
    description: 'Generate summaries using LLM',
    blockCount: 5,
    generates: ['DuplicateColumns', 'PromptBuilder', 'LLMChat', 'LLMParser', 'TextParser'],
    parameters: {
      summary_type: { type: 'select', label: 'Summary Type', options: ['detailed', 'atomic_facts', 'extractive'], default: 'detailed' },
      input_column: { type: 'text', label: 'Input Column', default: 'document' },
      output_column: { type: 'text', label: 'Output Column', default: 'summary' },
      max_tokens: { type: 'number', label: 'Max Tokens', default: 2048 },
      temperature: { type: 'number', label: 'Temperature', default: 0.7, min: 0, max: 2, step: 0.1 },
    }
  },
  {
    id: 'generation',
    name: 'Generation',
    icon: '❓',
    description: 'Generate question-answer pairs',
    blockCount: 4,
    generates: ['PromptBuilder', 'LLMChat', 'LLMParser', 'TextParser'],
    parameters: {
      num_qa_pairs: { type: 'number', label: 'Number of Q&A Pairs', default: 1, min: 1, max: 20 },
      input_column: { type: 'text', label: 'Input Column (Document)', default: 'document' },
      question_output: { type: 'text', label: 'Question Output Column', default: 'question' },
      answer_output: { type: 'text', label: 'Answer Output Column', default: 'response' },
      max_tokens: { type: 'number', label: 'Max Tokens', default: 2048 },
      temperature: { type: 'number', label: 'Temperature', default: 1.0, min: 0, max: 2, step: 0.1 },
    }
  },
  {
    id: 'evaluation',
    name: 'Evaluation',
    icon: '✓',
    description: 'Evaluate content quality with LLM',
    blockCount: 5,
    generates: ['PromptBuilder', 'LLMChat', 'LLMParser', 'TextParser', 'Filter'],
    parameters: {
      evaluation_type: { type: 'select', label: 'Evaluation Type', options: ['faithfulness', 'relevancy', 'quality'], default: 'faithfulness' },
      input_column_1: { type: 'text', label: 'First Input Column', default: 'document' },
      input_column_2: { type: 'text', label: 'Second Input Column', default: 'response' },
      output_explanation: { type: 'text', label: 'Explanation Column', default: 'evaluation_explanation' },
      output_judgment: { type: 'text', label: 'Judgment Column', default: 'evaluation_judgment' },
      filter_value: { type: 'text', label: 'Filter Value', default: 'YES' },
      max_tokens: { type: 'number', label: 'Max Tokens', default: 2048 },
    }
  },
];

