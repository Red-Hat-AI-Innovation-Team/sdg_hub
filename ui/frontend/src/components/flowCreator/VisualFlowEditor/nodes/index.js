export { default as LLMNodeConfig, LLMNodePreview } from './LLMNode';
export { default as ParserNodeConfig, ParserNodePreview } from './ParserNode';
export { default as EvalNodeConfig, EvalNodePreview } from './EvalNode';
export { default as TransformNodeConfig, TransformNodePreview } from './TransformNode';

import LLMNodeConfig from './LLMNode';
import ParserNodeConfig from './ParserNode';
import EvalNodeConfig from './EvalNode';
import TransformNodeConfig from './TransformNode';

/**
 * Map of all node configurations by type
 * 
 * Note: Prompt is now part of LLM node (PromptBuilder + LLMChat + Extractor)
 */
export const NODE_CONFIGS = {
  llm: LLMNodeConfig,
  parser: ParserNodeConfig,
  eval: EvalNodeConfig,
  transform: TransformNodeConfig,
};

/**
 * Get node configuration by type
 */
export const getNodeConfig = (type) => {
  return NODE_CONFIGS[type] || null;
};

/**
 * Get all node types as array
 */
export const getAllNodeTypes = () => {
  return Object.keys(NODE_CONFIGS);
};

export default NODE_CONFIGS;
