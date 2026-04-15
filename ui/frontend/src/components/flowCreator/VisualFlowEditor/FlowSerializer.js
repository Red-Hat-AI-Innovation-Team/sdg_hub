/**
 * Flow Serializer
 * 
 * Converts the visual graph representation to the YAML block array format
 * used by SDG Hub flows.
 */

import { NODE_TYPES } from './constants';
import { getNodeConfig } from './nodes';

/**
 * Get the primary output column(s) that a node produces
 * Used to derive downstream node input columns from edge connections
 */
export const getNodeOutputColumns = (node) => {
  const config = node.config || {};
  
  if (node.type === NODE_TYPES.LLM) {
    // LLM nodes produce a content column from the extractor block.
    // The name depends on the extractor's field_prefix or block_name:
    //   - If field_prefix is set (e.g. "topic_"): output is "topic_content"
    //   - If extractor has a custom block_name (e.g. "detailed_summary"): output is "detailed_summary_content"
    //   - Default: "extract_{llm_block_name}_content"
    const ext = config._extractor_block_config;
    if (ext?.field_prefix) {
      return [`${ext.field_prefix}content`];
    }
    if (ext?.block_name) {
      return [`${ext.block_name}_content`];
    }
    return [`extract_${config.block_name}_content`];
  } else if (node.type === NODE_TYPES.PARSER) {
    // Parser nodes produce their output_cols
    return Array.isArray(config.output_cols) ? config.output_cols : [config.output_cols];
  } else if (node.type === NODE_TYPES.TRANSFORM) {
    if (config.transform_type === 'duplicate' || config.transform_type === 'rename') {
      // Duplicate/rename produce the mapped values
      return Object.values(config.input_cols || config.column_mapping || {});
    } else if (config.transform_type === 'melt') {
      return config.output_cols || config.melt_output_cols || [];
    }
  } else if (node.type === NODE_TYPES.EVAL) {
    // Eval nodes produce filtered data, columns like {block_name}_judgment
    return [`${config.block_name}_judgment`, `${config.block_name}_explanation`];
  }
  
  // Fallback to config.output_cols
  if (config.output_cols) {
    return Array.isArray(config.output_cols) ? config.output_cols : [config.output_cols];
  }
  
  return [];
};

/**
 * Serialize visual flow to block array
 * 
 * @param {Array} nodes - Array of visual node objects
 * @param {Array} edges - Array of edge objects
 * @returns {Array} - Array of block configurations for flow.yaml
 */
export const serializeFlowToBlocks = (nodes, edges) => {
  // Build dependency map for topological sorting
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const incoming = new Map();
  const outgoing = new Map();

  nodes.forEach(node => {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  });

  edges.forEach(edge => {
    if (incoming.has(edge.target)) {
      incoming.get(edge.target).push(edge.source);
    }
    if (outgoing.has(edge.source)) {
      outgoing.get(edge.source).push(edge.target);
    }
  });

  // Topological sort using Kahn's algorithm
  const sortedNodes = topologicalSort(nodes, incoming, outgoing);

  // Build a map of what each node outputs (derived from current config, not stored input_cols)
  // This ensures renamed nodes produce the correct column names
  const nodeOutputMap = new Map();
  sortedNodes.forEach(node => {
    nodeOutputMap.set(node.id, getNodeOutputColumns(node));
  });

  // Convert each node to block(s), passing upstream info for column derivation
  const blocks = [];
  
  sortedNodes.forEach(node => {
    // Get the upstream node's outputs (if connected via edge)
    const upstreamNodeIds = incoming.get(node.id) || [];
    const upstreamOutputs = upstreamNodeIds.flatMap(id => nodeOutputMap.get(id) || []);
    
    const nodeBlocks = nodeToBlocks(node, nodeMap, edges, upstreamOutputs);
    blocks.push(...nodeBlocks);
  });

  return blocks;
};

/**
 * Topological sort of nodes
 */
const topologicalSort = (nodes, incoming, outgoing) => {
  const sorted = [];
  const inDegree = new Map();
  
  // Calculate in-degrees
  nodes.forEach(node => {
    inDegree.set(node.id, incoming.get(node.id)?.length || 0);
  });

  // Queue of nodes with no incoming edges
  const queue = nodes.filter(n => inDegree.get(n.id) === 0);

  while (queue.length > 0) {
    const node = queue.shift();
    sorted.push(node);

    // Reduce in-degree of neighbors
    const neighbors = outgoing.get(node.id) || [];
    neighbors.forEach(neighborId => {
      inDegree.set(neighborId, inDegree.get(neighborId) - 1);
      if (inDegree.get(neighborId) === 0) {
        const neighborNode = nodes.find(n => n.id === neighborId);
        if (neighborNode) {
          queue.push(neighborNode);
        }
      }
    });
  }

  // If not all nodes are sorted, there's a cycle - fall back to original order
  if (sorted.length !== nodes.length) {
    console.warn('Cycle detected in flow - using original order');
    return nodes;
  }

  return sorted;
};

/**
 * Convert a single visual node to block configuration(s)
 * 
 * @param {Object} node - Visual node object
 * @param {Map} nodeMap - Map of all nodes by ID
 * @param {Array} edges - Array of edges
 * @param {Array} upstreamOutputs - Output columns from upstream node(s)
 * @returns {Array} - Array of block configurations
 */
const nodeToBlocks = (node, nodeMap, edges, upstreamOutputs = []) => {
  const nodeConfig = getNodeConfig(node.type);
  const config = node.config || {};

  if (!nodeConfig) {
    console.warn(`Unknown node type: ${node.type}`);
    return [];
  }

  // For PARSER nodes connected to LLM nodes, derive input_cols from upstream
  // This ensures renamed LLM nodes automatically update downstream parser input columns
  let effectiveConfig = { ...config };
  
  if (node.type === NODE_TYPES.PARSER && upstreamOutputs.length > 0) {
    // Find the content column from upstream (e.g., extract_Friend_content)
    const contentCol = upstreamOutputs.find(col => col.endsWith('_content'));
    if (contentCol) {
      // Guard: if the parser already has input_cols set (from template) and
      // the derived column matches, no need to override. If they differ, only
      // override when the parser's value wasn't explicitly set in the template
      // (i.e., it was auto-derived in a previous round, indicated by _auto_derived_input_cols).
      const existingInputCols = config.input_cols;
      if (existingInputCols && existingInputCols === contentCol) {
        // Already correct, no change needed
        console.log(`Parser ${config.block_name}: input_cols "${existingInputCols}" already matches upstream, keeping as-is`);
      } else if (existingInputCols && !config._auto_derived_input_cols) {
        // Template value that differs from derived - trust the upstream computation
        // (which is now correct thanks to getNodeOutputColumns fix) but preserve
        // original as a fallback reference
        effectiveConfig = {
          ...config,
          input_cols: contentCol,
          _auto_derived_input_cols: true,
          _original_input_cols: config.input_cols,
        };
        console.log(`Parser ${config.block_name}: overriding template input_cols "${existingInputCols}" with upstream-derived "${contentCol}"`);
      } else {
        effectiveConfig = {
          ...config,
          input_cols: contentCol,
          _auto_derived_input_cols: true,
          _original_input_cols: config._original_input_cols || config.input_cols,
        };
        console.log(`Parser ${config.block_name}: derived input_cols "${contentCol}" from upstream`);
      }
    }
  }

  // Use the node's toBlockConfig method
  const blockResult = nodeConfig.toBlockConfig(effectiveConfig, effectiveConfig.prompt_config_path);

  // Handle both single blocks and arrays of blocks
  if (Array.isArray(blockResult)) {
    return blockResult;
  }
  
  return [blockResult];
};

/**
 * Deserialize block array to visual nodes and edges
 * Used when loading an existing flow for editing
 * 
 * @param {Array} blocks - Array of block configurations from flow.yaml
 * @returns {Object} - { nodes: Array, edges: Array }
 */
export const deserializeBlocksToVisualFlow = (blocks) => {
  const nodes = [];
  const edges = [];
  const blockToNodeMap = new Map();

  const startX = 100;
  let y = 100;
  const xSpacing = 250;
  const ySpacing = 140;
  let nodesInRow = 0;
  const maxNodesPerRow = 4;
  let currentRow = 0;

  // Group related blocks (e.g., PromptBuilder + LLM + Extractor = one visual node)
  const groupedBlocks = groupRelatedBlocks(blocks);

  groupedBlocks.forEach((group, index) => {
    const node = blockGroupToNode(group, index);
    
    if (node) {
      // Calculate position using serpentine (snake) layout
      // Even rows (0, 2, 4...): left to right
      // Odd rows (1, 3, 5...): right to left
      const isReversedRow = currentRow % 2 === 1;
      const col = isReversedRow ? (maxNodesPerRow - 1 - nodesInRow) : nodesInRow;
      node.position = { x: startX + col * xSpacing, y };
      nodesInRow++;
      
      if (nodesInRow >= maxNodesPerRow) {
        y += ySpacing;
        nodesInRow = 0;
        currentRow++;
      }

      nodes.push(node);

      // Map block names to node IDs for edge creation
      group.forEach(block => {
        blockToNodeMap.set(block.block_config?.block_name, node.id);
      });
    }
  });

  // Create edges based on data flow (input_cols references)
  nodes.forEach((targetNode, targetIndex) => {
    const inputCols = getNodeInputColumns(targetNode);
    
    inputCols.forEach(inputCol => {
      // Find the node that outputs this column
      const sourceNode = findNodeByOutput(nodes, inputCol);
      
      if (sourceNode && sourceNode.id !== targetNode.id) {
        // Check if edge already exists
        const edgeExists = edges.some(e => 
          e.source === sourceNode.id && e.target === targetNode.id
        );
        
        if (!edgeExists) {
          edges.push({
            id: `edge_${sourceNode.id}_${targetNode.id}`,
            source: sourceNode.id,
            target: targetNode.id,
          });
        }
      }
    });
  });

  return { nodes, edges };
};

/**
 * Group related blocks into logical units
 */
const groupRelatedBlocks = (blocks) => {
  const groups = [];
  let currentGroup = [];
  let i = 0;

  while (i < blocks.length) {
    const block = blocks[i];
    const blockType = block.block_type;

    // Check for Eval pattern (Prompt + LLM + Extractor + Parser + Filter)
    if (blockType === 'PromptBuilderBlock' && i + 4 < blocks.length) {
      const nextBlocks = blocks.slice(i + 1, i + 5);
      const isEvalPattern = (
        nextBlocks[0]?.block_type === 'LLMChatBlock' &&
        nextBlocks[1]?.block_type === 'LLMResponseExtractorBlock' &&
        (nextBlocks[2]?.block_type === 'TagParserBlock' || nextBlocks[2]?.block_type === 'RegexParserBlock') &&
        nextBlocks[3]?.block_type === 'ColumnValueFilterBlock'
      );

      if (isEvalPattern) {
        groups.push([block, ...nextBlocks]);
        i += 5;
        continue;
      }
    }

    // Check for LLM pattern (LLMChat + Extractor)
    if (blockType === 'LLMChatBlock' && i + 1 < blocks.length) {
      const nextBlock = blocks[i + 1];
      if (nextBlock?.block_type === 'LLMResponseExtractorBlock') {
        groups.push([block, nextBlock]);
        i += 2;
        continue;
      }
    }

    // Single block
    groups.push([block]);
    i++;
  }

  return groups;
};

/**
 * Convert a block group to a visual node
 */
const blockGroupToNode = (group, index) => {
  const primaryBlock = group[0];
  const blockType = primaryBlock.block_type;
  const blockConfig = primaryBlock.block_config || {};

  // Determine visual node type based on block pattern
  let nodeType;
  let config = {};

  if (group.length >= 5 && group[4]?.block_type === 'ColumnValueFilterBlock') {
    // Eval node pattern
    nodeType = NODE_TYPES.EVAL;
    config = {
      block_name: blockConfig.block_name?.replace('_prompt', '') || `eval_${index}`,
      input_cols: blockConfig.input_cols || [],
      filter_value: group[4].block_config?.filter_value || 'YES',
      filter_operation: group[4].block_config?.operation || 'eq',
      filter_dtype: group[4].block_config?.convert_dtype || null,
    };
    
    // Parse config from TagParserBlock or RegexParserBlock
    const parserBlock = group[3];
    if (parserBlock?.block_type === 'TagParserBlock' && parserBlock?.block_config?.start_tags) {
      config.start_tags = parserBlock.block_config.start_tags;
      config.end_tags = parserBlock.block_config.end_tags || [];
    }
    if (parserBlock?.block_type === 'RegexParserBlock') {
      config.parsing_pattern = parserBlock.block_config?.parsing_pattern || '';
    }
  } else if (blockType === 'PromptBuilderBlock') {
    nodeType = NODE_TYPES.PROMPT;
    config = {
      block_name: blockConfig.block_name,
      input_cols: blockConfig.input_cols || [],
      output_cols: blockConfig.output_cols,
      prompt_config_path: blockConfig.prompt_config_path,
      format_as_messages: blockConfig.format_as_messages,
    };
  } else if (blockType === 'LLMChatBlock') {
    nodeType = NODE_TYPES.LLM;
    config = {
      block_name: blockConfig.block_name,
      input_cols: blockConfig.input_cols,
      output_cols: blockConfig.output_cols,
      max_tokens: blockConfig.max_tokens,
      temperature: blockConfig.temperature,
      n: blockConfig.n,
      async_mode: blockConfig.async_mode,
      // Flag to skip PromptBuilder generation - when loaded from template,
      // the PromptBuilder is already a separate node
      _skipPromptBuilder: true,
    };
    
    // Preserve the original LLM block config for template flows
    // This ensures output column names are preserved exactly
    config._llm_block_config = {
      block_name: blockConfig.block_name,
      input_cols: blockConfig.input_cols,
      output_cols: blockConfig.output_cols,
      max_tokens: blockConfig.max_tokens,
      temperature: blockConfig.temperature,
      n: blockConfig.n,
      async_mode: blockConfig.async_mode,
    };
    
    // If this LLM block is paired with an extractor, preserve the extractor's config
    // This ensures that when we serialize back, we use the original block names
    if (group.length >= 2 && group[1]?.block_type === 'LLMResponseExtractorBlock') {
      const extractorConfig = group[1].block_config || {};
      config._extractor_block_config = {
        block_name: extractorConfig.block_name,
        input_cols: extractorConfig.input_cols,
        output_cols: extractorConfig.output_cols,
        field_prefix: extractorConfig.field_prefix,
        extract_content: extractorConfig.extract_content,
        extract_reasoning_content: extractorConfig.extract_reasoning_content,
        expand_lists: extractorConfig.expand_lists,
      };
    }
  } else if (blockType === 'TagParserBlock' || blockType === 'RegexParserBlock') {
    nodeType = NODE_TYPES.PARSER;
    config = {
      block_name: blockConfig.block_name,
      input_cols: blockConfig.input_cols,
      output_cols: Array.isArray(blockConfig.output_cols) 
        ? blockConfig.output_cols 
        : [blockConfig.output_cols],
      start_tags: blockConfig.start_tags || [],
      end_tags: blockConfig.end_tags || [],
      parsing_pattern: blockConfig.parsing_pattern,
    };
  } else if (blockType === 'DuplicateColumnsBlock') {
    nodeType = NODE_TYPES.TRANSFORM;
    config = {
      block_name: blockConfig.block_name,
      transform_type: 'duplicate',
      column_mapping: blockConfig.input_cols,
    };
  } else if (blockType === 'RenameColumnsBlock') {
    nodeType = NODE_TYPES.TRANSFORM;
    config = {
      block_name: blockConfig.block_name,
      transform_type: 'rename',
      column_mapping: blockConfig.input_cols,
    };
  } else if (blockType === 'MeltColumnsBlock') {
    nodeType = NODE_TYPES.TRANSFORM;
    config = {
      block_name: blockConfig.block_name,
      transform_type: 'melt',
      melt_input_cols: blockConfig.input_cols,
      melt_output_cols: blockConfig.output_cols,
    };
  } else {
    // Unknown block type - skip
    console.warn(`Unknown block type for visual conversion: ${blockType}`);
    return null;
  }

  return {
    id: `node_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
    type: nodeType,
    label: config.block_name || blockConfig.block_name || nodeType,
    config,
    configured: true,
    position: { x: 0, y: 0 }, // Will be set by caller
  };
};

/**
 * Get input column names for a node
 */
const getNodeInputColumns = (node) => {
  const config = node.config || {};
  const inputs = [];

  if (typeof config.input_cols === 'string') {
    inputs.push(config.input_cols);
  } else if (Array.isArray(config.input_cols)) {
    inputs.push(...config.input_cols);
  } else if (typeof config.input_cols === 'object') {
    inputs.push(...Object.keys(config.input_cols));
  }

  return inputs;
};

/**
 * Find a node that outputs a specific column
 */
const findNodeByOutput = (nodes, outputCol) => {
  for (const node of nodes) {
    const config = node.config || {};
    let outputs = [];

    if (typeof config.output_cols === 'string') {
      outputs = [config.output_cols];
    } else if (Array.isArray(config.output_cols)) {
      outputs = config.output_cols;
    }

    // Also check for column_mapping outputs (transform nodes)
    if (config.column_mapping) {
      outputs.push(...Object.values(config.column_mapping));
    }

    // Check for LLM extractor output pattern
    if (node.type === NODE_TYPES.LLM && config.block_name) {
      outputs.push(`extract_${config.block_name}_content`);
    }

    if (outputs.includes(outputCol)) {
      return node;
    }
  }

  return null;
};

/**
 * Compute required columns for the flow
 * 
 * Required columns are input columns that are NOT produced by any node in the flow.
 * These columns must come from the input dataset.
 * 
 * @param {Array} nodes - Array of visual node objects
 * @param {Array} edges - Array of edge objects
 * @returns {Array} - Array of column names required from the dataset
 */
export const computeRequiredColumns = (nodes, edges) => {
  // Collect all output columns produced by nodes in the flow
  const producedColumns = new Set();
  
  nodes.forEach(node => {
    const outputs = getNodeOutputColumns(node);
    outputs.forEach(col => producedColumns.add(col));
    
    // For LLM nodes, also add the user-specified output_cols (the final extracted column)
    if (node.type === NODE_TYPES.LLM && node.config?.output_cols) {
      // LLM nodes produce: {output_cols}_content (via extractor)
      producedColumns.add(`${node.config.output_cols}_content`);
    }
  });
  
  // Collect all input columns required by nodes
  const requiredInputs = new Set();
  
  nodes.forEach(node => {
    const config = node.config || {};
    
    // Get input columns based on node type
    if (node.type === NODE_TYPES.LLM || node.type === NODE_TYPES.PARSER || node.type === NODE_TYPES.EVAL) {
      // These nodes use input_cols array
      const inputs = Array.isArray(config.input_cols) 
        ? config.input_cols 
        : (config.input_cols ? [config.input_cols] : []);
      inputs.forEach(col => requiredInputs.add(col));
    } else if (node.type === NODE_TYPES.TRANSFORM) {
      // Transform nodes have different input patterns
      if (config.transform_type === 'duplicate' || config.transform_type === 'rename') {
        // column_mapping: { source: target }
        const mapping = config.column_mapping || config.input_cols || {};
        Object.keys(mapping).forEach(col => requiredInputs.add(col));
      } else if (config.transform_type === 'melt') {
        const meltInputs = config.melt_input_cols || config.input_cols || [];
        meltInputs.forEach(col => requiredInputs.add(col));
      }
    }
  });
  
  // Required columns = inputs that are NOT produced by any node
  const requiredColumns = [];
  requiredInputs.forEach(col => {
    if (!producedColumns.has(col)) {
      requiredColumns.push(col);
    }
  });
  
  // Sort for consistent ordering
  return requiredColumns.sort();
};

/**
 * Generate flow metadata from visual nodes
 */
export const generateFlowMetadata = (nodes, edges, existingMetadata = {}) => {
  // Determine flow type based on nodes present
  const hasEval = nodes.some(n => n.type === NODE_TYPES.EVAL);
  const nodeTypes = [...new Set(nodes.map(n => n.type))];

  const tags = [];
  if (nodes.some(n => n.type === NODE_TYPES.PROMPT)) tags.push('question-generation');
  if (hasEval) tags.push('evaluation');
  if (nodes.some(n => n.config?.transform_type === 'melt')) tags.push('data-transformation');

  return {
    name: existingMetadata.name || 'Custom Flow',
    description: existingMetadata.description || 'A custom flow created with the Visual Flow Builder',
    version: existingMetadata.version || '1.0.0',
    author: existingMetadata.author || 'SDG Hub User',
    tags: existingMetadata.tags || tags,
    license: existingMetadata.license || 'Apache-2.0',
    recommended_models: existingMetadata.recommended_models || {
      default: 'meta-llama/Llama-3.3-70B-Instruct',
      compatible: ['openai/gpt-oss-120b', 'microsoft/phi-4'],
    },
  };
};

export default {
  serializeFlowToBlocks,
  deserializeBlocksToVisualFlow,
  generateFlowMetadata,
  computeRequiredColumns,
};
