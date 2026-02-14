/**
 * Connection Validator
 * 
 * Validates whether two node types can be connected.
 * Enforces the flow logic based on the SDG Hub block types.
 * 
 * Node types (simplified):
 * - LLM: PromptBuilder + LLMChat + ResponseExtractor (combined)
 * - Parser: TextParserBlock (extract tagged sections)
 * - Eval: LLM + Parser + Filter (combined)
 * - Transform: Data manipulation blocks
 */

import { NODE_TYPES } from './constants';

/**
 * Connection rules matrix
 * Defines which source node types can connect to which target node types
 */
const ALL_NODE_TYPES = [NODE_TYPES.LLM, NODE_TYPES.PARSER, NODE_TYPES.EVAL, NODE_TYPES.TRANSFORM];

const CONNECTION_RULES = {
  // Any node type can connect to any other node type.
  // Data flows through as columns -- the user controls what's consumed via input_cols.
  [NODE_TYPES.LLM]: {
    validTargets: ALL_NODE_TYPES,
    message: 'LLM nodes can connect to any node type',
  },
  [NODE_TYPES.PARSER]: {
    validTargets: ALL_NODE_TYPES,
    message: 'Parser nodes can connect to any node type',
  },
  [NODE_TYPES.EVAL]: {
    validTargets: ALL_NODE_TYPES,
    message: 'Eval nodes can connect to any node type',
  },
  [NODE_TYPES.TRANSFORM]: {
    validTargets: ALL_NODE_TYPES,
    message: 'Transform nodes can connect to any node type',
  },
};

/**
 * Validate if a connection between two node types is allowed
 * 
 * @param {string} sourceType - The type of the source node
 * @param {string} targetType - The type of the target node
 * @returns {Object} - { valid: boolean, message: string }
 */
export const validateConnection = (sourceType, targetType) => {
  // Check if source type has rules defined
  const rules = CONNECTION_RULES[sourceType];
  
  if (!rules) {
    return {
      valid: false,
      message: `Unknown source node type: ${sourceType}`,
    };
  }

  // Check if target type is in the valid targets list
  if (rules.validTargets.includes(targetType)) {
    return {
      valid: true,
      message: 'Connection allowed',
    };
  }

  // Connection not allowed
  return {
    valid: false,
    message: `Invalid connection: ${rules.message}`,
  };
};

/**
 * Get valid target types for a given source type
 * 
 * @param {string} sourceType - The type of the source node
 * @returns {string[]} - Array of valid target node types
 */
export const getValidTargets = (sourceType) => {
  const rules = CONNECTION_RULES[sourceType];
  return rules?.validTargets || [];
};

/**
 * Get valid source types for a given target type
 * 
 * @param {string} targetType - The type of the target node
 * @returns {string[]} - Array of valid source node types
 */
export const getValidSources = (targetType) => {
  const validSources = [];
  
  Object.entries(CONNECTION_RULES).forEach(([sourceType, rules]) => {
    if (rules.validTargets.includes(targetType)) {
      validSources.push(sourceType);
    }
  });
  
  return validSources;
};

/**
 * Validate an entire flow graph for connection validity
 * 
 * @param {Array} nodes - Array of node objects
 * @param {Array} edges - Array of edge objects with source and target
 * @returns {Object} - { valid: boolean, errors: string[] }
 */
export const validateFlowConnections = (nodes, edges) => {
  const errors = [];
  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  edges.forEach(edge => {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);

    if (!sourceNode) {
      errors.push(`Edge ${edge.id}: Source node not found`);
      return;
    }

    if (!targetNode) {
      errors.push(`Edge ${edge.id}: Target node not found`);
      return;
    }

    const validation = validateConnection(sourceNode.type, targetNode.type);
    if (!validation.valid) {
      errors.push(`Edge from "${sourceNode.label || sourceNode.type}" to "${targetNode.label || targetNode.type}": ${validation.message}`);
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * Check if the flow has a valid structure (no cycles, proper entry/exit)
 * 
 * @param {Array} nodes - Array of node objects
 * @param {Array} edges - Array of edge objects
 * @returns {Object} - { valid: boolean, warnings: string[] }
 */
export const validateFlowStructure = (nodes, edges) => {
  const warnings = [];

  if (nodes.length === 0) {
    return { valid: false, warnings: ['Flow has no nodes'] };
  }

  // Build adjacency map
  const outgoing = new Map();
  const incoming = new Map();
  
  nodes.forEach(node => {
    outgoing.set(node.id, []);
    incoming.set(node.id, []);
  });

  edges.forEach(edge => {
    if (outgoing.has(edge.source)) {
      outgoing.get(edge.source).push(edge.target);
    }
    if (incoming.has(edge.target)) {
      incoming.get(edge.target).push(edge.source);
    }
  });

  // Check for entry points (nodes with no incoming edges)
  const entryNodes = nodes.filter(n => incoming.get(n.id)?.length === 0);
  if (entryNodes.length === 0) {
    warnings.push('Flow has no entry point (all nodes have incoming connections - possible cycle)');
  }

  // Check for exit points (nodes with no outgoing edges)
  const exitNodes = nodes.filter(n => outgoing.get(n.id)?.length === 0);
  if (exitNodes.length === 0 && nodes.length > 1) {
    warnings.push('Flow has no exit point (all nodes have outgoing connections - possible cycle)');
  }

  // Check for disconnected nodes
  const disconnectedNodes = nodes.filter(n => 
    incoming.get(n.id)?.length === 0 && 
    outgoing.get(n.id)?.length === 0 &&
    nodes.length > 1
  );
  if (disconnectedNodes.length > 0) {
    const names = disconnectedNodes.map(n => n.label || n.type).join(', ');
    warnings.push(`Disconnected nodes found: ${names}`);
  }

  // Simple cycle detection using DFS
  const visited = new Set();
  const recursionStack = new Set();

  const hasCycle = (nodeId) => {
    visited.add(nodeId);
    recursionStack.add(nodeId);

    const neighbors = outgoing.get(nodeId) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        if (hasCycle(neighbor)) return true;
      } else if (recursionStack.has(neighbor)) {
        return true;
      }
    }

    recursionStack.delete(nodeId);
    return false;
  };

  for (const node of nodes) {
    if (!visited.has(node.id)) {
      if (hasCycle(node.id)) {
        warnings.push('Flow contains a cycle - this may cause infinite loops');
        break;
      }
    }
  }

  return {
    valid: warnings.length === 0,
    warnings,
  };
};

/**
 * Get recommended next node types based on current node
 * 
 * @param {string} currentNodeType - The type of the current node
 * @returns {Array} - Array of { type, reason } objects
 */
export const getRecommendedNextNodes = (currentNodeType) => {
  const recommendations = [];
  const validTargets = getValidTargets(currentNodeType);

  validTargets.forEach(targetType => {
    switch (targetType) {
      case NODE_TYPES.LLM:
        recommendations.push({
          type: targetType,
          reason: 'Generate content with an LLM',
        });
        break;
      case NODE_TYPES.PARSER:
        recommendations.push({
          type: targetType,
          reason: 'Extract tagged sections from text',
        });
        break;
      case NODE_TYPES.EVAL:
        recommendations.push({
          type: targetType,
          reason: 'Evaluate quality and filter results',
        });
        break;
      case NODE_TYPES.TRANSFORM:
        recommendations.push({
          type: targetType,
          reason: 'Transform or reshape the data',
        });
        break;
    }
  });

  return recommendations;
};

export default {
  validateConnection,
  getValidTargets,
  getValidSources,
  validateFlowConnections,
  validateFlowStructure,
  getRecommendedNextNodes,
};
