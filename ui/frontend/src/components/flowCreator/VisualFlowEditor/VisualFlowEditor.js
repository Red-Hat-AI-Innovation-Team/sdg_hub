import React, { useState, useEffect, useCallback, useRef, useMemo, useReducer } from 'react';
import {
  Button,
  Toolbar,
  ToolbarContent,
  ToolbarItem,
  Title,
  Alert,
  AlertVariant,
  AlertActionCloseButton,
  Split,
  SplitItem,
  Tooltip,
  Spinner,
} from '@patternfly/react-core';
import {
  SaveIcon,
  PlayIcon,
  TrashIcon,
  CubesIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  SyncAltIcon,
  OutlinedQuestionCircleIcon,
} from '@patternfly/react-icons';

import NodeSidebar from './NodeSidebar';
import NodeConfigDrawer from './NodeConfigDrawer';
import GuidedTour, { TOUR_STORAGE_KEY } from './GuidedTour';
import { TestConfigModal, NodeIOModal } from '../TestRunner';
import { validateConnection } from './ConnectionValidator';
import { serializeFlowToBlocks, computeRequiredColumns, getNodeOutputColumns } from './FlowSerializer';
import { NODE_TYPES, NODE_TYPE_CONFIG, generateNodeId, generateEdgeId } from './constants';
import { flowTestAPI, workspaceAPI } from '../../../services/api';
import SimpleFlowCanvas from './SimpleFlowCanvas';

// Debounce utility
const debounce = (fn, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
};

/**
 * Extract tags from prompt text for auto-populating Parser nodes
 * Looks for [TAG] patterns and categorizes them as start/end tags
 * @param {string} text - The prompt text to analyze
 * @returns {{ startTags: string[], endTags: string[] }}
 */
const extractTagsFromPrompt = (text) => {
  if (!text) return { startTags: [], endTags: [] };
  
  // Find all [TAG] patterns (including [Start of X], [End of X], etc.)
  const tagPattern = /\[([^\]]+)\]/g;
  const matches = [...text.matchAll(tagPattern)];
  
  if (matches.length === 0) return { startTags: [], endTags: [] };
  
  // Extract unique tags
  const allTags = [...new Set(matches.map(m => m[0]))];
  
  const startTags = [];
  const endTags = [];
  
  // Categorize tags based on common patterns
  for (const tag of allTags) {
    const tagLower = tag.toLowerCase();
    
    // Check for explicit end patterns
    if (tagLower.includes('end') || 
        tagLower.includes('/') ||  // [/TAG] style
        tagLower.includes('stop') ||
        tagLower.includes('finish') ||
        tagLower.includes('close')) {
      endTags.push(tag);
    }
    // Check for explicit start patterns
    else if (tagLower.includes('start') || 
             tagLower.includes('begin') ||
             tagLower.includes('open')) {
      startTags.push(tag);
    }
    // For other tags, try to pair them
    else {
      // If we haven't categorized yet, add to start tags
      // User can adjust later
      startTags.push(tag);
    }
  }
  
  // If we have equal numbers of uncategorized tags and no explicit end tags,
  // try to pair them (e.g., [QUESTION] [ANSWER] -> start: [QUESTION], end: [ANSWER])
  if (endTags.length === 0 && startTags.length >= 2) {
    // Move half to end tags (assumes tags come in pairs in order)
    const halfPoint = Math.ceil(startTags.length / 2);
    const movedToEnd = startTags.splice(halfPoint);
    endTags.push(...movedToEnd);
  }
  
  return { startTags, endTags };
};

// CSS styles for test animations
const testAnimationStyles = `
@keyframes nodePulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0, 102, 204, 0.4); }
  50% { box-shadow: 0 0 20px 10px rgba(0, 102, 204, 0.6); }
}

@keyframes nodeSuccess {
  0% { transform: scale(1); }
  50% { transform: scale(1.02); }
  100% { transform: scale(1); }
}

@keyframes edgeFlow {
  from { stroke-dashoffset: 24; }
  to { stroke-dashoffset: 0; }
}

.node-test-running {
  animation: nodePulse 1s infinite;
  border-color: #0066cc !important;
}

.node-test-complete {
  animation: nodeSuccess 0.3s ease-out;
  border-color: #3e8635 !important;
}

.node-test-error {
  border-color: #c9190b !important;
}

.edge-test-active {
  stroke: #0066cc !important;
  stroke-dasharray: 8 4;
  animation: edgeFlow 0.5s linear infinite;
}

.edge-test-complete {
  stroke: #3e8635 !important;
}
`;

// Re-export for backwards compatibility
export { NODE_TYPES, NODE_TYPE_CONFIG } from './constants';

// --- Test execution state reducer ---
const testInitialState = {
  isTestRunning: false,
  testResults: {},
  activeTestNode: null,
  activeTestEdges: [],
  completedTestNodes: [],
  showTestConfig: false,
  showNodeIOModal: false,
  selectedNodeForIO: null,
  requiredTestColumns: [],
};

const testReducer = (state, action) => {
  switch (action.type) {
    case 'START_TEST':
      return {
        ...state,
        isTestRunning: true,
        testResults: {},
        activeTestNode: null,
        activeTestEdges: [],
        completedTestNodes: [],
      };
    case 'STOP_TEST':
      return { ...state, isTestRunning: false };
    case 'TEST_COMPLETE':
      return {
        ...state,
        isTestRunning: false,
        activeTestNode: null,
        activeTestEdges: [],
      };
    case 'SET_ACTIVE_TEST_NODE':
      return { ...state, activeTestNode: action.nodeId };
    case 'SET_ACTIVE_TEST_EDGES':
      return { ...state, activeTestEdges: action.edgeIds };
    case 'NODE_TEST_COMPLETE':
      return {
        ...state,
        completedTestNodes: [...state.completedTestNodes, action.nodeId],
        testResults: { ...state.testResults, [action.nodeId]: action.result },
        activeTestNode: null,
      };
    case 'NODE_TEST_ERROR':
      return {
        ...state,
        testResults: { ...state.testResults, [action.nodeId]: { status: 'error', error: action.error } },
        activeTestNode: null,
      };
    case 'SHOW_TEST_CONFIG':
      return { ...state, showTestConfig: true, requiredTestColumns: action.columns };
    case 'HIDE_TEST_CONFIG':
      return { ...state, showTestConfig: false };
    case 'SHOW_NODE_IO_MODAL':
      return { ...state, showNodeIOModal: true, selectedNodeForIO: action.node };
    case 'HIDE_NODE_IO_MODAL':
      return { ...state, showNodeIOModal: false, selectedNodeForIO: null };
    case 'CLEAR_TEST_RESULTS':
      return {
        ...state,
        testResults: {},
        completedTestNodes: [],
        activeTestNode: null,
        activeTestEdges: [],
      };
    default:
      return state;
  }
};

// --- Workspace state reducer ---
const workspaceInitialState = {
  workspaceId: null,
  workspacePath: null,
  isSyncing: false,
  syncError: null,
};

const workspaceReducer = (state, action) => {
  switch (action.type) {
    case 'SET_WORKSPACE':
      return { ...state, workspaceId: action.workspaceId, workspacePath: action.workspacePath };
    case 'CLEAR_WORKSPACE':
      return { ...state, workspaceId: null, workspacePath: null };
    case 'SET_SYNCING':
      return { ...state, isSyncing: action.isSyncing };
    case 'SET_SYNC_ERROR':
      return { ...state, syncError: action.error };
    case 'SYNC_START':
      return { ...state, isSyncing: true, syncError: null };
    case 'SYNC_SUCCESS':
      return { ...state, isSyncing: false };
    case 'SYNC_ERROR':
      return { ...state, isSyncing: false, syncError: action.error };
    default:
      return state;
  }
};

/**
 * Main Visual Flow Editor Component
 */
const VisualFlowEditor = ({ 
  initialFlow, 
  onSave, 
  onBack, 
  onDraftChange,
  existingFlowName,
  isEditMode,
  isBlankStart = false, // True when starting from "Create Custom Flow" (blank)
}) => {
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [showConfigDrawer, setShowConfigDrawer] = useState(false);
  const [nodes, setNodes] = useState([]);
  
  // Guided tour state
  const [forceStartTour, setForceStartTour] = useState(false);
  const [isTourActive, setIsTourActive] = useState(false);
  const [sidebarActiveTab, setSidebarActiveTab] = useState('node-library');
  const isFirstVisit = useMemo(() => {
    // Only show tour for blank starts that haven't completed the tour
    if (!isBlankStart) return false;
    const tourCompleted = localStorage.getItem(TOUR_STORAGE_KEY);
    return !tourCompleted;
  }, [isBlankStart]);
  
  // Handler to manually start the tour
  const handleStartGuidedTour = useCallback(() => {
    localStorage.removeItem(TOUR_STORAGE_KEY);
    setForceStartTour(true);
  }, []);
  const [edges, setEdges] = useState([]);
  const [flowMetadata, setFlowMetadata] = useState(() => {
    const metadata = initialFlow?.metadata || {};
    // Ensure required_columns is at top level (extract from dataset_requirements if needed)
    if (!metadata.required_columns || metadata.required_columns.length === 0) {
      const reqCols = metadata.dataset_requirements?.required_columns || [];
      if (reqCols.length > 0) {
        return { ...metadata, required_columns: reqCols };
      }
    }
    return metadata;
  });
  const [error, setError] = useState(null);
  const [alertVariant, setAlertVariant] = useState(AlertVariant.danger);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [sourceFlowName, setSourceFlowName] = useState(initialFlow?.sourceFlowName || null);
  // Original blocks from template - used for testing to preserve exact block configuration
  const [originalBlocks, setOriginalBlocks] = useState(null);
  const originalBlocksRef = useRef(null);
  // Keep ref in sync with state so debounced callbacks can access current value
  useEffect(() => { originalBlocksRef.current = originalBlocks; }, [originalBlocks]);
  
  // Workspace state (live flow editing)
  const [workspace, dispatchWorkspace] = useReducer(workspaceReducer, workspaceInitialState);
  const { workspaceId, workspacePath, isSyncing, syncError } = workspace;
  const workspaceInitializedRef = useRef(false);

  // Test execution state
  const [testState, dispatchTest] = useReducer(testReducer, testInitialState);
  const { isTestRunning, testResults, activeTestNode, activeTestEdges,
          completedTestNodes, showTestConfig, showNodeIOModal,
          selectedNodeForIO, requiredTestColumns } = testState;
  
  // Inject test animation styles
  useEffect(() => {
    const styleId = 'test-animation-styles';
    if (!document.getElementById(styleId)) {
      const style = document.createElement('style');
      style.id = styleId;
      style.textContent = testAnimationStyles;
      document.head.appendChild(style);
    }
    return () => {
      const style = document.getElementById(styleId);
      if (style) style.remove();
    };
  }, []);

  /**
   * Initialize nodes and edges from initial flow
   * Always update when visualNodes in initialFlow changes
   */
  const lastVisualNodesRef = React.useRef(null);
  const isInitializedRef = React.useRef(false);
  
  /**
   * Generate sequential edges for nodes if edges are missing
   * This ensures edges are always created when loading flows with nodes but no edges
   */
  const generateSequentialEdges = (nodesList) => {
    const generatedEdges = [];
    for (let i = 0; i < nodesList.length - 1; i++) {
      generatedEdges.push({
        id: `edge_${nodesList[i].id}_to_${nodesList[i + 1].id}`,
        source: nodesList[i].id,
        target: nodesList[i + 1].id,
        type: 'default',
      });
    }
    return generatedEdges;
  };

  useEffect(() => {
    const visualNodesCount = initialFlow?.visualNodes?.length || 0;
    const visualEdgesCount = initialFlow?.visualEdges?.length || 0;
    
    // Create a signature to detect actual changes in the visual nodes data
    const newNodesSignature = initialFlow?.visualNodes 
      ? JSON.stringify(initialFlow.visualNodes.map(n => ({ id: n.id, type: n.type, config: n.config })))
      : '';
    
    console.log('VisualFlowEditor useEffect: visualNodes:', visualNodesCount, 'visualEdges:', visualEdgesCount, 'current nodes:', nodes.length, 'isInitialized:', isInitializedRef.current);
    
    // Only set nodes if initialFlow has visual data AND either:
    // 1. We haven't initialized yet, OR
    // 2. The actual visual nodes data changed (not just count)
    if (visualNodesCount > 0 && (!isInitializedRef.current || newNodesSignature !== lastVisualNodesRef.current)) {
      console.log('VisualFlowEditor: Setting nodes from initialFlow:', visualNodesCount, 'nodes');
      setNodes(initialFlow.visualNodes);
      
      // Build a set of valid node IDs for quick lookup
      const validNodeIds = new Set(initialFlow.visualNodes.map(n => n.id));
      
      // Check if existing edges reference valid node IDs
      const edgesAreValid = initialFlow.visualEdges && initialFlow.visualEdges.length > 0 &&
        initialFlow.visualEdges.every(edge => 
          validNodeIds.has(edge.source) && validNodeIds.has(edge.target)
        );
      
      // If edges are missing, incomplete, or reference invalid node IDs, regenerate them
      const expectedEdgeCount = visualNodesCount - 1;
      if (!edgesAreValid || visualEdgesCount < expectedEdgeCount) {
        console.log('VisualFlowEditor: Regenerating edges (had', visualEdgesCount, 'expected', expectedEdgeCount, ', valid:', edgesAreValid, ')');
        const regeneratedEdges = generateSequentialEdges(initialFlow.visualNodes);
        setEdges(regeneratedEdges);
      } else {
        setEdges(initialFlow.visualEdges);
      }
      
      lastVisualNodesRef.current = newNodesSignature;
      isInitializedRef.current = true;
    }
  }, [initialFlow?.visualNodes, initialFlow?.visualEdges]);

  /**
   * Sync flowMetadata when initialFlow.metadata changes (for editing/cloning)
   * This ensures required_columns are properly extracted from dataset_requirements
   */
  useEffect(() => {
    if (initialFlow?.metadata) {
      const metadata = initialFlow.metadata;
      // Extract required_columns from dataset_requirements if not at top level
      let reqCols = metadata.required_columns || [];
      if ((!reqCols || reqCols.length === 0) && metadata.dataset_requirements?.required_columns?.length > 0) {
        reqCols = metadata.dataset_requirements.required_columns;
      }
      
      console.log('VisualFlowEditor: Syncing flowMetadata from initialFlow, required_columns:', reqCols);
      
      setFlowMetadata(prev => ({
        ...prev,
        ...metadata,
        required_columns: reqCols,
      }));
    }
  }, [initialFlow?.metadata]);

  /**
   * Track unsaved changes and notify parent
   * Only notify after user has made changes (not on initial mount/initialization)
   */
  const hasUserChangedRef = React.useRef(false);
  const prevNodesLengthRef = React.useRef(nodes.length);
  const prevEdgesLengthRef = React.useRef(edges.length);
  
  useEffect(() => {
    // Detect if this is a user-driven change vs initialization
    // User changes: adding/removing nodes, changing edges after initialization
    const isUserChange = isInitializedRef.current && (
      // Node count changed after initialization
      (nodes.length !== prevNodesLengthRef.current) ||
      // Edge count changed after initialization (user connected/disconnected nodes)
      (edges.length !== prevEdgesLengthRef.current) ||
      // Or we've already marked as having user changes
      hasUserChangedRef.current
    );
    
    if (isUserChange) {
      hasUserChangedRef.current = true;
    }
    
    prevNodesLengthRef.current = nodes.length;
    prevEdgesLengthRef.current = edges.length;
    
    // Only notify parent of changes after user has interacted
    // This prevents overwriting parent state during initialization
    if (onDraftChange && hasUserChangedRef.current) {
      onDraftChange({
        visualNodes: nodes,
        visualEdges: edges,
        metadata: flowMetadata,
      });
    }
    // Mark as having unsaved changes if there are any nodes
    setHasUnsavedChanges(nodes.length > 0);
  }, [nodes, edges, flowMetadata]);

  /**
   * Recompute required columns when nodes change
   * Required columns are input columns that aren't produced by any node in the flow
   */
  useEffect(() => {
    if (!isInitializedRef.current || nodes.length === 0) {
      return;
    }
    
    const computedRequiredCols = computeRequiredColumns(nodes, edges);
    
    // Only update if the computed columns are different from current
    const currentRequiredCols = flowMetadata.required_columns || [];
    const hasChanged = 
      computedRequiredCols.length !== currentRequiredCols.length ||
      computedRequiredCols.some(col => !currentRequiredCols.includes(col)) ||
      currentRequiredCols.some(col => !computedRequiredCols.includes(col));
    
    if (hasChanged) {
      console.log('VisualFlowEditor: Auto-computed required_columns:', computedRequiredCols);
      setFlowMetadata(prev => ({
        ...prev,
        required_columns: computedRequiredCols,
      }));
    }
  }, [nodes, edges]);

  /**
   * Initialize workspace on mount (for new flows, cloning, or editing)
   * Note: Template loading creates workspace in handleLoadFlowTemplate
   */
  useEffect(() => {
    const initWorkspace = async () => {
      // Already have a workspace
      if (workspaceId) {
        return;
      }
      
      // Determine source flow name for workspace creation
      // Priority: sourceFlowName (cloning) > originalFlowName (editing) > existingFlowName prop
      const sourceFlow = initialFlow?.sourceFlowName || initialFlow?.originalFlowName || existingFlowName;
      
      console.log('initWorkspace: Determining source flow:', {
        'initialFlow?.sourceFlowName': initialFlow?.sourceFlowName,
        'initialFlow?.originalFlowName': initialFlow?.originalFlowName,
        'existingFlowName': existingFlowName,
        'computed sourceFlow': sourceFlow,
        'isCloning': initialFlow?.isCloning,
        'isEditing': initialFlow?.isEditing,
        'hasVisualNodes': initialFlow?.visualNodes?.length > 0,
        'alreadyInitialized': workspaceInitializedRef.current,
      });
      
      // For cloning or editing: wait until we have the source flow name
      // Don't create empty workspace prematurely
      if ((initialFlow?.isCloning || initialFlow?.isEditing) && !sourceFlow) {
        console.log('Waiting for source flow name (cloning/editing mode)...');
        return; // Wait for next effect cycle when sourceFlow is available
      }
      
      // For flows with visual nodes but no source: also wait for source flow name
      // (This happens when initialFlow is loading and visualNodes arrive before originalFlowName)
      if (initialFlow?.visualNodes?.length > 0 && !sourceFlow && !workspaceInitializedRef.current) {
        console.log('Waiting for source flow name (has visual nodes)...');
        return; // Wait for next effect cycle
      }
      
      // Now we can proceed with workspace creation
      if (workspaceInitializedRef.current) {
        return;
      }
      workspaceInitializedRef.current = true;
      
      try {
        // For cloning or editing existing flows: create workspace from source to copy prompt files
        if (sourceFlow && (initialFlow?.isCloning || initialFlow?.isEditing || initialFlow?.visualNodes?.length > 0)) {
          console.log('Creating workspace from source flow:', sourceFlow, '(isCloning:', initialFlow?.isCloning, ', isEditing:', initialFlow?.isEditing, ')');
          const result = await workspaceAPI.create(sourceFlow);
          dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: result.workspace_id, workspacePath: result.workspace_path });
          setSourceFlowName(sourceFlow);
          console.log('Workspace created from source:', result.workspace_id);
          return;
        }
        
        // For new flows (no source and no initial visual nodes): create empty workspace
        if (!initialFlow?.visualNodes?.length) {
          console.log('Creating empty workspace for new flow');
          const result = await workspaceAPI.create(null);
          dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: result.workspace_id, workspacePath: result.workspace_path });
          console.log('Empty workspace created:', result.workspace_id);
        }
      } catch (err) {
        console.error('Failed to create workspace:', err);
        dispatchWorkspace({ type: 'SET_SYNC_ERROR', error: 'Failed to initialize workspace' });
      }
    };
    
    initWorkspace();
  }, [initialFlow?.visualNodes?.length, initialFlow?.isCloning, initialFlow?.isEditing, initialFlow?.sourceFlowName, initialFlow?.originalFlowName, existingFlowName, workspaceId]);

  /**
   * Debounced sync to workspace
   * Triggers when nodes, edges, or metadata change
   */
  const syncToWorkspaceRef = useRef(null);
  
  useEffect(() => {
    // Create debounced sync function
    syncToWorkspaceRef.current = debounce(async (currentNodes, currentEdges, currentMetadata, wsId) => {
      if (!wsId) {
        console.log('No workspace ID, skipping sync');
        return;
      }
      
      try {
        dispatchWorkspace({ type: 'SYNC_START' });
        
        // Use original template blocks if available (no user modifications),
        // otherwise re-serialize from visual nodes
        const blocks = originalBlocksRef.current || serializeFlowToBlocks(currentNodes, currentEdges);
        
        // Update flow.yaml in workspace
        await workspaceAPI.updateFlow(wsId, currentMetadata, blocks);
        
        console.log(`Synced ${blocks.length} blocks to workspace ${wsId}${originalBlocksRef.current ? ' (using original template blocks)' : ''}`);
        dispatchWorkspace({ type: 'SYNC_SUCCESS' });
      } catch (err) {
        console.error('Failed to sync to workspace:', err);
        dispatchWorkspace({ type: 'SYNC_ERROR', error: 'Failed to sync changes' });
      }
    }, 1000); // 1 second debounce
    
    return () => {
      // Cleanup: no need to clear timeout as debounce handles it
    };
  }, []);

  // Trigger sync when flow changes (only if workspace exists and user has made changes)
  useEffect(() => {
    if (workspaceId && hasUserChangedRef.current && nodes.length > 0) {
      syncToWorkspaceRef.current?.(nodes, edges, flowMetadata, workspaceId);
    }
  }, [nodes, edges, flowMetadata, workspaceId]);

  /**
   * Cleanup workspace on unmount if not saved
   */
  useEffect(() => {
    return () => {
      // Cleanup workspace on unmount if it exists and has unsaved changes
      const wsId = workspaceId;
      if (wsId && hasUnsavedChanges) {
        console.log('Cleaning up abandoned workspace:', wsId);
        workspaceAPI.delete(wsId).catch(err => {
          console.error('Failed to cleanup workspace:', err);
        });
      }
    };
  }, []); // Empty deps - only runs on unmount

  /**
   * Add a new node to the canvas and open config drawer automatically
   */
  const handleAddNode = useCallback((nodeType, position = { x: 100, y: 100 }) => {
    const newNode = {
      id: generateNodeId(),
      type: nodeType,
      label: NODE_TYPE_CONFIG[nodeType]?.label || nodeType,
      config: getDefaultNodeConfig(nodeType),
      position,
      configured: false,
    };

    setNodes(prev => [...prev, newNode]);
    // Clear original blocks since flow is being modified
    setOriginalBlocks(null);
    // Auto-select the new node and open the config drawer
    setSelectedNodeId(newNode.id);
    setShowConfigDrawer(true);
    
    return newNode;
  }, []);

  /**
   * Handle drop on canvas
   */
  const handleCanvasDrop = useCallback((e) => {
    e.preventDefault();
    const nodeType = e.dataTransfer.getData('nodeType');
    
    if (nodeType && NODE_TYPE_CONFIG[nodeType]) {
      // Get drop position relative to canvas
      const canvasRect = e.currentTarget.getBoundingClientRect();
      const position = {
        x: e.clientX - canvasRect.left - 100, // Center the node
        y: e.clientY - canvasRect.top - 40,
      };
      
      handleAddNode(nodeType, position);
    }
  }, [handleAddNode]);

  /**
   * Handle drag over canvas (allow drop)
   */
  const handleCanvasDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  /**
   * Get default configuration for a node type
   */
  const getDefaultNodeConfig = (nodeType) => {
    switch (nodeType) {
      case NODE_TYPES.LLM:
        // LLM node combines: PromptBuilder + LLMChat + ResponseExtractor
        return {
          block_name: '',
          input_cols: [],
          output_cols: '',
          system_message: 'You are a helpful AI assistant.',
          user_message: '',
          max_tokens: 2048,
          temperature: 0.7,
          n: 1,
          async_mode: true,
        };
      case NODE_TYPES.PARSER:
        return {
          block_name: '',
          input_cols: '',
          output_cols: [],
          start_tags: [],
          end_tags: [],
          parsing_pattern: '',
        };
      case NODE_TYPES.EVAL:
        // Eval node combines: LLM + Parser + Filter
        return {
          block_name: '',
          input_cols: [],
          output_cols: [],
          system_message: '',
          user_message: '',
          start_tags: ['[Start of Explanation]', '[Start of Answer]'],
          end_tags: ['[End of Explanation]', '[End of Answer]'],
          filter_value: 'YES',
          filter_operation: 'eq',
        };
      case NODE_TYPES.TRANSFORM:
        return {
          block_name: '',
          transform_type: 'duplicate', // duplicate, rename, melt
          input_cols: {},
          output_cols: [],
        };
      default:
        return {};
    }
  };

  /**
   * Update node configuration
   */
  const handleUpdateNodeConfig = useCallback(async (nodeId, newConfig) => {
    // Find the node to check its type
    const nodeToUpdate = nodes.find(n => n.id === nodeId);
    
    // Build the updated node to compute its output columns
    const updatedSourceNode = {
      ...nodeToUpdate,
      config: { ...nodeToUpdate?.config, ...newConfig },
    };
    const sourceOutputCols = getNodeOutputColumns(updatedSourceNode);
    const primaryOutputCol = sourceOutputCols.length > 0 ? sourceOutputCols[sourceOutputCols.length - 1] : '';
    
    // Find the downstream node connected via an outgoing edge
    const outgoingEdge = edges.find(e => e.source === nodeId);
    const downstreamNodeId = outgoingEdge?.target;

    setNodes(prev => prev.map(node => {
      if (node.id === nodeId) {
        return {
          ...node,
          config: { ...node.config, ...newConfig },
          configured: true,
          label: newConfig.block_name || node.label,
        };
      }
      // Auto-update downstream node's input_cols when source config changes
      if (node.id === downstreamNodeId && primaryOutputCol) {
        const updatedConfig = { ...node.config };
        if (node.type === NODE_TYPES.PARSER) {
          updatedConfig.input_cols = primaryOutputCol;
        } else if (node.type === NODE_TYPES.EVAL || node.type === NODE_TYPES.LLM) {
          updatedConfig.input_cols = [primaryOutputCol];
        }
        return { ...node, config: updatedConfig };
      }
      return node;
    }));
    // Clear original blocks since flow is being modified
    setOriginalBlocks(null);
    setShowConfigDrawer(false);
    
    // For LLM and EVAL nodes with prompt content, sync the prompt file to workspace
    // These nodes have system_message and user_message, and use prompt_config_path for the file
    const nodeType = nodeToUpdate?.type;
    const hasPromptContent = newConfig.user_message || newConfig.system_message;
    
    if (workspaceId && (nodeType === 'llm' || nodeType === 'eval') && hasPromptContent) {
      try {
        // Determine the prompt filename based on node type:
        // - LLM nodes: LLMNode.js expects `${block_name}_prompt.yaml` (line 210)
        // - EVAL nodes: EvalNode.js expects `${block_name}.yaml` (line 204)
        // For renamed nodes or existing paths, extract/generate the appropriate filename
        let promptFilename;
        const originalBlockName = nodeToUpdate?.config?._llm_block_config?.block_name;
        const blockNameChanged = originalBlockName && originalBlockName !== newConfig.block_name;
        
        if (blockNameChanged) {
          // Node was renamed - save to new file matching the new name
          // Use _prompt suffix only for LLM nodes (matches LLMNode.js line 210)
          promptFilename = nodeType === 'llm' 
            ? `${newConfig.block_name}_prompt.yaml`
            : `${newConfig.block_name}.yaml`;
        } else if (newConfig.prompt_config_path) {
          // Extract just the filename from potentially absolute path
          promptFilename = newConfig.prompt_config_path.split('/').pop();
        } else {
          // New node without existing prompt
          // LLM nodes: use _prompt suffix (LLMNode.js line 210)
          // EVAL nodes: use plain block_name (EvalNode.js line 204)
          promptFilename = nodeType === 'llm'
            ? `${newConfig.block_name}_prompt.yaml`
            : `${newConfig.block_name}.yaml`;
        }
        
        const promptContent = [
          ...(newConfig.system_message ? [{ role: 'system', content: newConfig.system_message }] : []),
          ...(newConfig.user_message ? [{ role: 'user', content: newConfig.user_message }] : []),
        ];
        
        const result = await workspaceAPI.updatePrompt(workspaceId, promptFilename, { messages: promptContent });
        console.log(`${nodeType.toUpperCase()} prompt synced to workspace:`, result.full_prompt_path, '(file:', promptFilename, ')');
      } catch (err) {
        console.error(`Failed to sync ${nodeType} prompt to workspace:`, err);
        dispatchWorkspace({ type: 'SET_SYNC_ERROR', error: 'Failed to sync prompt' });
      }
    }
  }, [nodes, workspaceId]);

  /**
   * Inject config into a node (used by guided tour)
   */
  const handleInjectNodeConfig = useCallback((nodeId, config) => {
    setNodes(prev => prev.map(node => {
      if (node.id === nodeId) {
        return {
          ...node,
          config: { ...node.config, ...config },
          configured: true, // Mark as configured so required columns can be computed
        };
      }
      return node;
    }));
  }, []);

  /**
   * Delete a node and its connected edges
   */
  const handleDeleteNode = useCallback((nodeId) => {
    setNodes(prev => {
      const newNodes = prev.filter(node => node.id !== nodeId);
      return newNodes; // Allow empty array
    });
    setEdges(prev => prev.filter(edge => 
      edge.source !== nodeId && edge.target !== nodeId
    ));
    // Clear original blocks since flow is being modified
    setOriginalBlocks(null);
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
      setShowConfigDrawer(false);
    }
  }, [selectedNodeId]);

  /**
   * Handle keyboard events for node deletion
   */
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Delete or Backspace key to delete selected node
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNodeId) {
        // Don't delete if user is typing in an input field
        const activeElement = document.activeElement;
        const isInputField = activeElement && (
          activeElement.tagName === 'INPUT' ||
          activeElement.tagName === 'TEXTAREA' ||
          activeElement.isContentEditable
        );
        
        if (!isInputField) {
          e.preventDefault();
          handleDeleteNode(selectedNodeId);
        }
      }
      
      // Escape key to deselect node
      if (e.key === 'Escape') {
        setSelectedNodeId(null);
        setShowConfigDrawer(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, handleDeleteNode]);

  /**
   * Add an edge between two nodes
   * Auto-populates Parser node config when connected to LLM node
   */
  const handleAddEdge = useCallback((sourceId, targetId) => {
    const sourceNode = nodes.find(n => n.id === sourceId);
    const targetNode = nodes.find(n => n.id === targetId);

    if (!sourceNode || !targetNode) return false;

    // Validate connection
    const validation = validateConnection(sourceNode.type, targetNode.type);
    if (!validation.valid) {
      setError(validation.message);
      setTimeout(() => setError(null), 3000);
      return false;
    }

    // Check for duplicate edge
    const exists = edges.some(e => e.source === sourceId && e.target === targetId);
    if (exists) return false;

    // Enforce one edge per node: each node can have at most 1 outgoing and 1 incoming edge
    const sourceHasOutgoing = edges.some(e => e.source === sourceId);
    if (sourceHasOutgoing) return false;
    const targetHasIncoming = edges.some(e => e.target === targetId);
    if (targetHasIncoming) return false;

    const newEdge = {
      id: generateEdgeId(),
      source: sourceId,
      target: targetId,
    };

    // Auto-populate target node's input_cols from source node's output columns.
    // Uses getNodeOutputColumns (from FlowSerializer) for the REAL column names.
    // ALWAYS overwrites input_cols so the wiring stays correct.
    const sourceOutputCols = getNodeOutputColumns(sourceNode);
    const primaryOutputCol = sourceOutputCols.length > 0 ? sourceOutputCols[sourceOutputCols.length - 1] : '';

    if (primaryOutputCol) {
      setNodes(prev => prev.map(node => {
        if (node.id !== targetId) return node;
        const currentConfig = node.config || {};
        const updatedConfig = { ...currentConfig };

        if (targetNode.type === NODE_TYPES.PARSER) {
          updatedConfig.input_cols = primaryOutputCol;
          // Also auto-fill tags from LLM prompt if connecting LLM → Parser
          if (sourceNode.type === NODE_TYPES.LLM) {
            const promptText = `${sourceNode.config?.system_message || ''} ${sourceNode.config?.user_message || ''}`;
            const { startTags, endTags } = extractTagsFromPrompt(promptText);
            if ((!currentConfig.start_tags || currentConfig.start_tags.length === 0) && startTags.length > 0) {
              updatedConfig.start_tags = startTags;
            }
            if ((!currentConfig.end_tags || currentConfig.end_tags.length === 0) && endTags.length > 0) {
              updatedConfig.end_tags = endTags;
            }
          }
        } else if (targetNode.type === NODE_TYPES.EVAL) {
          updatedConfig.input_cols = [primaryOutputCol];
        } else if (targetNode.type === NODE_TYPES.LLM) {
          updatedConfig.input_cols = [primaryOutputCol];
        } else if (targetNode.type === NODE_TYPES.TRANSFORM) {
          // Transform nodes use input_cols as object for duplicate/rename, but set it for melt
          if (currentConfig.transform_type === 'melt') {
            updatedConfig.melt_input_cols = [primaryOutputCol];
          }
        }

        return { ...node, config: updatedConfig };
      }));
    }

    setEdges(prev => [...prev, newEdge]);
    // Clear original blocks since flow is being modified
    setOriginalBlocks(null);
    return true;
  }, [nodes, edges]);

  /**
   * Delete an edge
   */
  const handleDeleteEdge = useCallback((edgeId) => {
    setEdges(prev => prev.filter(edge => edge.id !== edgeId));
    // Clear original blocks since flow is being modified
    setOriginalBlocks(null);
  }, []);

  /**
   * Load a flow template onto the canvas
   * @param {Object} templateData - The template data to load
   * @param {boolean} isUpdate - If true, skip confirmation (used for prompt updates)
   */
  const handleLoadFlowTemplate = useCallback(async (templateData, isUpdate = false) => {
    // Confirm if there are existing nodes (unless this is an update to existing template)
    if (nodes.length > 0 && !isUpdate) {
      const confirm = window.confirm(
        'Loading a template will replace your current nodes. Continue?'
      );
      if (!confirm) return;
    }

    // Create workspace from template (unless this is just a prompt update)
    if (!isUpdate && templateData.sourceFlowName) {
      try {
        // Delete existing workspace if any
        if (workspaceId) {
          await workspaceAPI.delete(workspaceId).catch(() => {});
        }
        
        // Create new workspace from template
        console.log('Creating workspace from template:', templateData.sourceFlowName);
        const wsResult = await workspaceAPI.create(templateData.sourceFlowName);
        dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: wsResult.workspace_id, workspacePath: wsResult.workspace_path });
        workspaceInitializedRef.current = true;
        console.log('Workspace created:', wsResult.workspace_id);
        
        // Use blocks from workspace (they have full_prompt_path set)
        if (wsResult.blocks && wsResult.blocks.length > 0) {
          setOriginalBlocks(wsResult.blocks);
        }
      } catch (err) {
        console.error('Failed to create workspace from template:', err);
        dispatchWorkspace({ type: 'SET_SYNC_ERROR', error: 'Failed to create workspace' });
        // Continue anyway - we can still show the visual nodes
      }
    }

    // Set the visual nodes and edges from the template
    setNodes(templateData.visualNodes || []);
    setEdges(templateData.visualEdges || []);
    
    // Update metadata if provided (skip on updates to avoid overwriting user edits)
    if (templateData.metadata && !isUpdate) {
      // Extract required_columns from dataset_requirements if present
      const requiredColumns = templateData.metadata.dataset_requirements?.required_columns 
        || templateData.metadata.required_columns 
        || [];
      
      setFlowMetadata(prev => ({
        ...prev,
        ...templateData.metadata,
        name: templateData.metadata.name ? `${templateData.metadata.name} (Copy)` : prev.name,
        // Ensure required_columns is set at top level for MetadataFormModal
        required_columns: requiredColumns,
      }));
    }
    
    // Track source flow for prompt file copying (when saving)
    if (templateData.sourceFlowName && !isUpdate) {
      setSourceFlowName(templateData.sourceFlowName);
    }
    
    // Store original blocks from template for testing (if not already set from workspace)
    // This ensures we use the exact block configuration without re-serializing
    if (templateData.originalBlocks && !isUpdate && !originalBlocks) {
      setOriginalBlocks(templateData.originalBlocks);
    }

    // Clear selection
    setSelectedNodeId(null);
    setShowConfigDrawer(false);
  }, [nodes.length, workspaceId, originalBlocks]);

  /**
   * Handle node selection
   */
  const handleNodeSelect = useCallback((nodeId) => {
    setSelectedNodeId(nodeId);
    if (nodeId) {
      setShowConfigDrawer(true);
    }
  }, []);

  /**
   * Handle save flow
   */
  const handleSave = useCallback(async () => {
    try {
      // Use original template blocks if available, otherwise re-serialize
      const blocks = originalBlocks || serializeFlowToBlocks(nodes, edges);
      
      // If we have a workspace, finalize it (rename to permanent name)
      if (workspaceId) {
        // First, sync the latest changes to workspace
        await workspaceAPI.updateFlow(workspaceId, flowMetadata, blocks);
        
        // Then finalize the workspace
        const flowName = flowMetadata.name || 'Custom Flow';
        console.log('Finalizing workspace:', workspaceId, 'as', flowName);
        const finalizeResult = await workspaceAPI.finalize(workspaceId, flowName);
        
        // Clear workspace state (it's now a permanent flow)
        dispatchWorkspace({ type: 'CLEAR_WORKSPACE' });
        setHasUnsavedChanges(false);
        
        // Notify parent with the finalized flow info
        if (onSave) {
          await onSave({
            metadata: flowMetadata,
            blocks,
            visualNodes: nodes,
            visualEdges: edges,
            sourceFlowName: sourceFlowName,
            savedPath: finalizeResult.flow_path,
            savedDir: finalizeResult.flow_dir,
          });
        }
        
        console.log('Flow saved successfully:', finalizeResult.flow_path);
      } else {
        // No workspace - use the old save method (for backwards compatibility)
        const completeFlow = {
          metadata: flowMetadata,
          blocks,
          visualNodes: nodes,
          visualEdges: edges,
          sourceFlowName: sourceFlowName,
        };

        if (onSave) {
          await onSave(completeFlow);
          setHasUnsavedChanges(false);
        }
      }
    } catch (err) {
      console.error('Failed to save flow:', err);
      setError('Failed to save flow: ' + err.message);
    }
  }, [nodes, edges, flowMetadata, onSave, sourceFlowName, workspaceId, originalBlocks]);

  /**
   * Get the topologically sorted order of nodes
   */
  const getNodeOrder = useCallback(() => {
    const nodeOrder = [];
    const visited = new Set();
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    
    // Find nodes with no incoming edges (source nodes)
    const sourceNodeIds = nodes
      .filter(n => !edges.some(e => e.target === n.id))
      .map(n => n.id);
    
    // BFS to get topological order
    const queue = [...sourceNodeIds];
    while (queue.length > 0) {
      const nodeId = queue.shift();
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);
      
      const node = nodeMap.get(nodeId);
      if (node) nodeOrder.push(node);
      
      // Add connected nodes
      edges.filter(e => e.source === nodeId).forEach(e => {
        if (!visited.has(e.target)) {
          queue.push(e.target);
        }
      });
    }
    
    // Add any unvisited nodes
    nodes.forEach(n => {
      if (!visited.has(n.id)) {
        nodeOrder.push(n);
      }
    });
    
    return nodeOrder;
  }, [nodes, edges]);

  /**
   * Map block index to node ID
   * Since one visual node can generate multiple blocks, we need to track this
   */
  const buildBlockToNodeMap = useCallback(() => {
    const blockToNode = {};
    const blocks = serializeFlowToBlocks(nodes, edges);
    const nodeOrder = getNodeOrder();
    
    let blockIndex = 0;
    nodeOrder.forEach(node => {
      // Count how many blocks this node generates
      const nodeConfig = NODE_TYPE_CONFIG[node.type];
      let blocksPerNode = 1;
      
      // LLM nodes generate 2-3 blocks depending on whether PromptBuilder is skipped
      // (2 if loaded from template with _skipPromptBuilder, 3 otherwise)
      // Eval nodes generate 5 blocks
      if (node.type === 'llm') {
        blocksPerNode = node.config?._skipPromptBuilder ? 2 : 3;
      } else if (node.type === 'eval') {
        blocksPerNode = 5;
      }
      
      // Map each block index to this node
      for (let i = 0; i < blocksPerNode; i++) {
        blockToNode[blockIndex + i] = node.id;
      }
      blockIndex += blocksPerNode;
    });
    
    return blockToNode;
  }, [nodes, edges, getNodeOrder]);

  /**
   * Handle test button click
   */
  const handleTestClick = useCallback(() => {
    // Validate we have nodes to test
    if (nodes.length === 0) {
      setError('Add nodes to the flow before testing');
      return;
    }
    
    // Check if all nodes are configured
    const unconfiguredNodes = nodes.filter(n => !n.configured);
    if (unconfiguredNodes.length > 0) {
      setError(`Configure all nodes before testing. ${unconfiguredNodes.length} node(s) not configured.`);
      return;
    }

    // Check for orphan nodes (nodes with no edges) - only when there are multiple nodes
    if (nodes.length > 1) {
      const connectedNodeIds = new Set();
      edges.forEach(e => {
        connectedNodeIds.add(e.source);
        connectedNodeIds.add(e.target);
      });
      const orphanNodes = nodes.filter(n => !connectedNodeIds.has(n.id));
      if (orphanNodes.length > 0) {
        const orphanNames = orphanNodes.map(n => n.config?.block_name || n.label || 'Unnamed').join(', ');
        setError(`All nodes must be connected. Found ${orphanNodes.length} disconnected node(s): ${orphanNames}. Connect them or remove them before testing.`);
        return;
      }
    }
    
    // Get required columns from multiple sources (in priority order):
    // 1. Flow metadata required_columns (from template or user config)
    // 2. Flow metadata dataset_requirements.required_columns
    // 3. Initial flow's metadata (for cloned flows)
    // 4. First node's input_cols (fallback)
    let requiredCols = [];
    
    console.log('handleTestClick: flowMetadata:', flowMetadata);
    console.log('handleTestClick: initialFlow?.metadata:', initialFlow?.metadata);
    
    // Check flow metadata first
    if (flowMetadata.required_columns && flowMetadata.required_columns.length > 0) {
      requiredCols = flowMetadata.required_columns;
      console.log('handleTestClick: Using flowMetadata.required_columns:', requiredCols);
    } else if (flowMetadata.dataset_requirements?.required_columns?.length > 0) {
      requiredCols = flowMetadata.dataset_requirements.required_columns;
      console.log('handleTestClick: Using flowMetadata.dataset_requirements.required_columns:', requiredCols);
    } else if (initialFlow?.metadata?.required_columns?.length > 0) {
      // Check initialFlow metadata (for cloned flows where flowMetadata might not be synced)
      requiredCols = initialFlow.metadata.required_columns;
      console.log('handleTestClick: Using initialFlow.metadata.required_columns:', requiredCols);
    } else if (initialFlow?.metadata?.dataset_requirements?.required_columns?.length > 0) {
      requiredCols = initialFlow.metadata.dataset_requirements.required_columns;
      console.log('handleTestClick: Using initialFlow.metadata.dataset_requirements.required_columns:', requiredCols);
    } else {
      // Fallback: extract from first node's input_cols
      const nodeOrder = getNodeOrder();
      const firstNode = nodeOrder[0];
      
      console.log('handleTestClick: Fallback - checking first node:', firstNode?.config);
      
      if (firstNode && firstNode.config) {
        const inputCols = firstNode.config.input_cols;
        if (Array.isArray(inputCols)) {
          requiredCols = inputCols;
        } else if (typeof inputCols === 'string' && inputCols) {
          requiredCols = [inputCols];
        }
        console.log('handleTestClick: Using first node input_cols:', requiredCols);
      }
    }
    
    console.log('handleTestClick: Final requiredCols:', requiredCols);
    dispatchTest({ type: 'SHOW_TEST_CONFIG', columns: requiredCols });
  }, [nodes, getNodeOrder, flowMetadata, initialFlow]);

  /**
   * Handle running the test
   */
  const handleRunTest = useCallback(async (testConfig) => {
    dispatchTest({ type: 'HIDE_TEST_CONFIG' });
    dispatchTest({ type: 'START_TEST' });
    setError(null);
    setAlertVariant(AlertVariant.danger);
    
    try {
      // ALWAYS sync to workspace before running test to ensure latest changes are saved
      // This is critical because the test loads blocks from the workspace
      let currentWorkspaceId = workspaceId;
      
      // Ensure we have a valid workspace (it may have been finalized/deleted)
      if (currentWorkspaceId) {
        try {
          dispatchWorkspace({ type: 'SET_SYNCING', isSyncing: true });
          const blocks = originalBlocks || serializeFlowToBlocks(nodes, edges);
          await workspaceAPI.updateFlow(currentWorkspaceId, flowMetadata, blocks);
          console.log(`Force-synced ${blocks.length} blocks to workspace before test${originalBlocks ? ' (using original template blocks)' : ''}`);
          dispatchWorkspace({ type: 'SET_SYNCING', isSyncing: false });
        } catch (syncErr) {
          console.warn('Workspace sync failed (may have been finalized). Creating new workspace...', syncErr);
          dispatchWorkspace({ type: 'SET_SYNCING', isSyncing: false });
          // Workspace is gone (finalized/deleted). Create a fresh one and sync.
          try {
            const wsResult = await workspaceAPI.create();
            currentWorkspaceId = wsResult.workspace_id;
            dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: wsResult.workspace_id, workspacePath: wsResult.workspace_path });
            workspaceInitializedRef.current = true;
            // Now sync prompt files for all LLM/eval nodes
            for (const node of nodes) {
              if ((node.type === 'llm' || node.type === 'eval') && node.config) {
                const cfg = node.config;
                if (cfg.user_message || cfg.system_message) {
                  const promptFilename = node.type === 'llm'
                    ? `${cfg.block_name}_prompt.yaml`
                    : `${cfg.block_name}.yaml`;
                  const promptContent = [
                    ...(cfg.system_message ? [{ role: 'system', content: cfg.system_message }] : []),
                    ...(cfg.user_message ? [{ role: 'user', content: cfg.user_message }] : []),
                  ];
                  await workspaceAPI.updatePrompt(currentWorkspaceId, promptFilename, promptContent).catch(() => {});
                }
              }
            }
            // Sync the flow
            const blocks = serializeFlowToBlocks(nodes, edges);
            await workspaceAPI.updateFlow(currentWorkspaceId, flowMetadata, blocks);
            console.log('Created new workspace and synced all nodes');
          } catch (createErr) {
            console.error('Failed to create new workspace:', createErr);
          }
        }
      } else {
        // No workspace at all -- create one
        try {
          const wsResult = await workspaceAPI.create();
          currentWorkspaceId = wsResult.workspace_id;
          dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: wsResult.workspace_id, workspacePath: wsResult.workspace_path });
          workspaceInitializedRef.current = true;
          const blocks = serializeFlowToBlocks(nodes, edges);
          await workspaceAPI.updateFlow(currentWorkspaceId, flowMetadata, blocks);
          // Sync prompts
          for (const node of nodes) {
            if ((node.type === 'llm' || node.type === 'eval') && node.config) {
              const cfg = node.config;
              if (cfg.user_message || cfg.system_message) {
                const promptFilename = node.type === 'llm'
                  ? `${cfg.block_name}_prompt.yaml`
                  : `${cfg.block_name}.yaml`;
                const promptContent = [
                  ...(cfg.system_message ? [{ role: 'system', content: cfg.system_message }] : []),
                  ...(cfg.user_message ? [{ role: 'user', content: cfg.user_message }] : []),
                ];
                await workspaceAPI.updatePrompt(currentWorkspaceId, promptFilename, promptContent).catch(() => {});
              }
            }
          }
        } catch (err) {
          console.error('Failed to create workspace for test:', err);
        }
      }
      
      // Serialize current visual nodes to blocks for node mapping
      // Note: The actual test will load blocks from the workspace (which was just synced)
      const blocks = serializeFlowToBlocks(nodes, edges);
      const nodeOrder = getNodeOrder();
      
      // Build block-to-node mapping from current visual nodes
      // Since we sync to workspace before test, blocks match visual state
      let blockToNodeMap = buildBlockToNodeMap();
      const nodeBlockCounts = {};
      const nodeCompletedBlocks = {};
      const localCompletedNodes = new Set();
      const nodeAccumulatedNewCols = {};
      const nodeLastNonEmptyOutput = {};
      
      nodeOrder.forEach(n => {
        const llmBlockCount = n.config?._skipPromptBuilder ? 2 : 3;
        nodeBlockCounts[n.id] = n.type === 'llm' ? llmBlockCount : n.type === 'eval' ? 5 : 1;
        nodeCompletedBlocks[n.id] = 0;
        nodeAccumulatedNewCols[n.id] = [];
        nodeLastNonEmptyOutput[n.id] = null;
      });
      
      // Connect to SSE endpoint
      // If we have a workspace, pass its ID so the backend loads from workspace files
      await flowTestAPI.testStepByStep({
        blocks,
        modelConfig: testConfig.modelConfig,
        sampleData: testConfig.sampleData,
        workspaceId: currentWorkspaceId,  // Backend will load from workspace if provided
      }, (event) => {
        console.log('Test event:', event);
        
        if (event.type === 'block_start') {
          const nodeId = blockToNodeMap[event.block_index];
          if (nodeId && !localCompletedNodes.has(nodeId)) {
            dispatchTest({ type: 'SET_ACTIVE_TEST_NODE', nodeId });
            
            // Activate incoming edges
            const incomingEdges = edges.filter(e => e.target === nodeId).map(e => e.id);
            dispatchTest({ type: 'SET_ACTIVE_TEST_EDGES', edgeIds: incomingEdges });
          }
        } 
        else if (event.type === 'block_complete') {
          const nodeId = blockToNodeMap[event.block_index];
          if (nodeId) {
            nodeCompletedBlocks[nodeId]++;
            
            // Accumulate new_columns from every block in this node.
            // For multi-block nodes (eval/llm), each sub-block may add
            // different columns; we collect them all so the UI can show
            // the complete set rather than only the last block's.
            if (event.new_columns && event.new_columns.length > 0) {
              nodeAccumulatedNewCols[nodeId] = [
                ...(nodeAccumulatedNewCols[nodeId] || []),
                ...event.new_columns.filter(
                  col => !(nodeAccumulatedNewCols[nodeId] || []).includes(col)
                ),
              ];
            }
            
            // Track the last block_complete event that had non-empty output.
            // For eval nodes the final block is a filter that may empty the
            // dataset; we want the UI to still show the pre-filter data so
            // the user can inspect the evaluation results.
            if (!event.skipped && event.output_data && event.output_data.length > 0) {
              nodeLastNonEmptyOutput[nodeId] = {
                output_data: event.output_data,
                output_columns: event.output_columns,
                output_rows: event.output_rows,
              };
            }
            
            // If all blocks for this node are complete, mark node as complete
            if (nodeCompletedBlocks[nodeId] >= nodeBlockCounts[nodeId]) {
              localCompletedNodes.add(nodeId);
              
              // Use accumulated new_columns from all blocks in this node
              const accumulatedNewCols = nodeAccumulatedNewCols[nodeId] || [];
              
              // If the final block's output is empty but we have earlier
              // non-empty output, use the earlier data so users can still
              // inspect evaluation results even when the filter removed all rows.
              const lastGoodOutput = nodeLastNonEmptyOutput[nodeId];
              const finalOutputData = (event.output_data && event.output_data.length > 0)
                ? event.output_data
                : (lastGoodOutput?.output_data || event.output_data);
              const finalOutputCols = (event.output_data && event.output_data.length > 0)
                ? event.output_columns
                : (lastGoodOutput?.output_columns || event.output_columns);
              const finalOutputRows = (event.output_data && event.output_data.length > 0)
                ? event.output_rows
                : (lastGoodOutput?.output_rows || event.output_rows);
              
              // Store the test result and mark node complete (also clears activeTestNode)
              dispatchTest({ type: 'NODE_TEST_COMPLETE', nodeId, result: {
                  status: 'complete',
                  input_data: event.input_data,
                  output_data: finalOutputData,
                  input_columns: event.input_columns,
                  output_columns: finalOutputCols,
                  new_columns: accumulatedNewCols.length > 0 ? accumulatedNewCols : event.new_columns,
                  execution_time_ms: event.execution_time_ms,
                  input_rows: event.input_rows,
                  output_rows: finalOutputRows,
              }});
            }
          }
        }
        else if (event.type === 'block_error') {
          const nodeId = blockToNodeMap[event.block_index];
          if (nodeId) {
            dispatchTest({ type: 'NODE_TEST_ERROR', nodeId, error: event.error });
          }
        }
        else if (event.type === 'filter_empty') {
          // A filter/parser block removed all samples (normal in single-sample test mode).
          // Mark all remaining (not-yet-completed) nodes as "skipped" so the user
          // can see they were never executed and understand why.
          console.warn('Filter emptied dataset:', event.message);
          setAlertVariant(AlertVariant.warning);
          setError('A block produced 0 output rows. Check the input and output of each node before proceeding.');

          // Mark remaining nodes as skipped
          nodeOrder.forEach(n => {
            if (!localCompletedNodes.has(n.id)) {
              localCompletedNodes.add(n.id);
              dispatchTest({ type: 'NODE_TEST_COMPLETE', nodeId: n.id, result: {
                status: 'skipped',
                input_data: [],
                output_data: [],
                input_columns: [],
                output_columns: [],
                new_columns: [],
                execution_time_ms: 0,
                input_rows: 0,
                output_rows: 0,
                skipped_reason: 'Previous block removed all samples',
              }});
            }
          });
        }
        else if (event.type === 'test_complete') {
          dispatchTest({ type: 'TEST_COMPLETE' });
          if (event.filter_emptied && !event.message?.includes('failed')) {
            // Filter emptied -- warning was already shown by filter_empty event
            setAlertVariant(prev => prev || AlertVariant.warning);
          } else if (!event.filter_emptied) {
            // All blocks completed successfully -- show success
            setAlertVariant(AlertVariant.success);
            setError('Test completed successfully. All blocks produced output.');
          }
        }
        else if (event.type === 'test_error') {
          setError(`Test failed: ${event.message}`);
          dispatchTest({ type: 'TEST_COMPLETE' });
        }
      });
    } catch (err) {
      console.error('Test error:', err);
      setError('Test failed: ' + err.message);
      dispatchTest({ type: 'STOP_TEST' });
    }
  }, [nodes, edges, buildBlockToNodeMap, getNodeOrder, workspaceId, flowMetadata, originalBlocks]);

  /**
   * Handle node click during/after test to show I/O modal
   */
  const handleNodeClickForIO = useCallback((nodeId) => {
    const nodeResult = testResults[nodeId];
    if (nodeResult) {
      const node = nodes.find(n => n.id === nodeId);
      dispatchTest({ type: 'SHOW_NODE_IO_MODAL', node });
    }
  }, [testResults, nodes]);

  /**
   * Clear test results
   */
  const clearTestResults = useCallback(() => {
    dispatchTest({ type: 'CLEAR_TEST_RESULTS' });
  }, []);

  /**
   * Handle clear all nodes
   */
  const handleClearAll = useCallback(async () => {
    if (nodes.length === 0) return;
    
    // Skip confirmation during tour for smoother experience
    const confirmed = isTourActive || window.confirm('Are you sure you want to remove all nodes? This cannot be undone.');
    if (confirmed) {
      // Clear visual state
      setNodes([]);
      setEdges([]);
      setSelectedNodeId(null);
      setShowConfigDrawer(false);
      
      // Clear flow metadata and original blocks so stale template data doesn't persist
      setOriginalBlocks(null);
      setSourceFlowName(null);
      setHasUnsavedChanges(false);
      setError(null);
      setFlowMetadata({ name: '', description: '', version: '1.0.0', required_columns: [] });
      
      // Stop draft auto-saving from firing with stale data
      hasUserChangedRef.current = false;
      prevNodesLengthRef.current = 0;
      prevEdgesLengthRef.current = 0;
      
      // Notify parent that the draft is now empty so it clears session/draft data
      if (onDraftChange) {
        onDraftChange({ visualNodes: [], visualEdges: [], metadata: {}, cleared: true });
      }
      
      // Clear test results
      dispatchTest({ type: 'CLEAR_TEST_RESULTS' });
      
      // Delete the old workspace and create a fresh empty one
      // so the next template load or save starts clean
      if (workspaceId) {
        try {
          await workspaceAPI.delete(workspaceId).catch(() => {});
          const wsResult = await workspaceAPI.create();
          dispatchWorkspace({ type: 'SET_WORKSPACE', workspaceId: wsResult.workspace_id, workspacePath: wsResult.workspace_path });
          workspaceInitializedRef.current = true;
        } catch (err) {
          console.warn('Failed to reset workspace on clear:', err);
        }
      }
    }
  }, [nodes.length, isTourActive, workspaceId]);

  /**
   * Handle node position change from canvas drag (memoized for React.memo)
   */
  const handleNodePositionChange = useCallback((nodeId, position) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId ? { ...node, position } : node
    ));
  }, []);

  /**
   * Handle edge deletion from canvas (without clearing originalBlocks)
   */
  const handleCanvasDeleteEdge = useCallback((edgeId) => {
    setEdges(prev => prev.filter(e => e.id !== edgeId));
  }, []);

  /**
   * Get the selected node
   */
  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Header Toolbar - Fixed at top */}
      <Toolbar style={{ flexShrink: 0 }}>
        <ToolbarContent>
          <ToolbarItem>
            <Title headingLevel="h2" size="xl">
              Visual Flow Builder
              {isEditMode && flowMetadata?.name && (
                <span style={{ fontSize: '1rem', fontWeight: 'normal', color: '#6a6e73', marginLeft: '12px' }}>
                  - Editing: {flowMetadata.name}
                </span>
              )}
            </Title>
          </ToolbarItem>
          {/* Workspace sync status indicator */}
          {workspaceId && (
            <ToolbarItem>
              <Tooltip content={isSyncing ? "Syncing changes to workspace..." : syncError ? syncError : "Changes auto-saved to workspace"}>
                <span style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '6px',
                  fontSize: '12px',
                  color: syncError ? '#c9190b' : isSyncing ? '#0066cc' : '#3e8635',
                  padding: '4px 8px',
                  background: syncError ? '#fce9e9' : isSyncing ? '#e7f1fa' : '#e8f5e3',
                  borderRadius: '4px',
                }}>
                  {isSyncing ? (
                    <><SyncAltIcon style={{ animation: 'spin 1s linear infinite' }} /> Syncing...</>
                  ) : syncError ? (
                    <><ExclamationCircleIcon /> Sync Error</>
                  ) : (
                    <><CheckCircleIcon /> Auto-saved</>
                  )}
                </span>
              </Tooltip>
            </ToolbarItem>
          )}
          <ToolbarItem align={{ default: 'alignRight' }}>
            <Split hasGutter>
              <SplitItem>
                <Tooltip content="Start the interactive guide to learn how to build flows">
                  <Button 
                    variant="link" 
                    icon={<OutlinedQuestionCircleIcon />}
                    onClick={handleStartGuidedTour}
                  >
                    Interactive Guide
                  </Button>
                </Tooltip>
              </SplitItem>
              <SplitItem>
                <Tooltip content="Remove all nodes from canvas">
                  <Button 
                    variant="secondary" 
                    icon={<TrashIcon />}
                    onClick={handleClearAll}
                    isDisabled={nodes.length === 0}
                    data-tour="clear-all-button"
                  >
                    Clear All
                  </Button>
                </Tooltip>
              </SplitItem>
              <SplitItem>
                <Tooltip content={isTestRunning ? "Test in progress..." : Object.keys(testResults).length > 0 ? "Run test again (clears previous results)" : "Test flow with sample data"}>
                  <Button 
                    variant="secondary" 
                    icon={isTestRunning ? <Spinner size="sm" /> : <PlayIcon />}
                    onClick={() => {
                      if (Object.keys(testResults).length > 0) {
                        clearTestResults();
                      }
                      handleTestClick();
                    }}
                    isDisabled={isTestRunning || nodes.length === 0}
                    data-tour="test-button"
                  >
                    {isTestRunning ? 'Testing...' : 'Test'}
                  </Button>
                </Tooltip>
              </SplitItem>
              <SplitItem>
                <Tooltip content={hasUnsavedChanges ? "Save flow" : "No changes to save"}>
                  <Button 
                    variant="primary" 
                    icon={<SaveIcon />}
                    onClick={handleSave}
                    isDisabled={nodes.length === 0}
                  >
                    Save
                  </Button>
                </Tooltip>
              </SplitItem>
            </Split>
          </ToolbarItem>
        </ToolbarContent>
      </Toolbar>

      {/* Error/Info Alert */}
      {error && (
        <Alert 
          variant={alertVariant} 
          isInline 
          title={error}
          actionClose={<AlertActionCloseButton onClose={() => { setError(null); setAlertVariant(AlertVariant.danger); }} />}
          style={{ margin: '0.5rem 1rem' }}
        />
      )}

      {/* Main Editor Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>
        {/* Left Sidebar - Node Palette */}
        <NodeSidebar 
          onAddNode={handleAddNode} 
          onLoadFlowTemplate={handleLoadFlowTemplate}
          onTabChange={setSidebarActiveTab}
        />

        {/* Center - Visual Canvas */}
        <div 
          style={{ 
            flex: 1, 
            position: 'relative', 
            background: '#fafafa',
            backgroundImage: 'radial-gradient(#d2d2d2 1px, transparent 1px)',
            backgroundSize: '20px 20px',
            overflow: 'auto',
          }}
          onDrop={handleCanvasDrop}
          onDragOver={handleCanvasDragOver}
        >
          <SimpleFlowCanvas
            nodes={nodes}
            edges={edges}
            selectedNodeId={selectedNodeId}
            onNodeSelect={handleNodeSelect}
            onNodePositionChange={handleNodePositionChange}
            onAddEdge={handleAddEdge}
            onDeleteNode={handleDeleteNode}
            onDeleteEdge={handleCanvasDeleteEdge}
            // Test execution props
            isTestRunning={isTestRunning}
            testResults={testResults}
            activeTestNode={activeTestNode}
            activeTestEdges={activeTestEdges}
            completedTestNodes={completedTestNodes}
            onNodeClickForIO={handleNodeClickForIO}
          />
          
          {/* Empty state */}
          {nodes.length === 0 && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              color: '#6a6e73',
              pointerEvents: 'none',
            }}>
              <CubesIcon style={{ fontSize: '48px', marginBottom: '16px', color: '#6a6e73' }} />
              <div style={{ fontSize: '18px', fontWeight: 500, marginBottom: '8px' }}>
                Drag nodes here to start building
              </div>
              <div style={{ fontSize: '14px' }}>
                Drag nodes from the left panel or double-click to add
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar - Node Config Drawer */}
        {showConfigDrawer && selectedNode && (
          <NodeConfigDrawer
            key={`${selectedNode.id}-${JSON.stringify(selectedNode.config?.system_message || '').slice(0, 20)}`}
            node={selectedNode}
            onClose={() => setShowConfigDrawer(false)}
            onSave={(config) => handleUpdateNodeConfig(selectedNode.id, config)}
            onDelete={() => handleDeleteNode(selectedNode.id)}
            existingFlowName={existingFlowName}
            workspaceId={workspaceId}
          />
        )}
      </div>

      {/* Test Configuration Modal */}
      <TestConfigModal
        isOpen={showTestConfig}
        onClose={() => dispatchTest({ type: 'HIDE_TEST_CONFIG' })}
        onRunTest={handleRunTest}
        requiredColumns={requiredTestColumns}
        isRunning={isTestRunning}
      />

      {/* Node I/O Modal */}
      <NodeIOModal
        isOpen={showNodeIOModal}
        onClose={() => {
          // After closing test results, open the node config drawer
          const nodeForConfig = selectedNodeForIO;
          dispatchTest({ type: 'HIDE_NODE_IO_MODAL' });
          if (nodeForConfig) {
            setSelectedNodeId(nodeForConfig.id);
            setShowConfigDrawer(true);
          }
        }}
        nodeData={selectedNodeForIO}
        testResult={selectedNodeForIO ? testResults[selectedNodeForIO.id] : null}
      />

      {/* Guided Tour (shown on first visit for blank starts, or when manually triggered) */}
      <GuidedTour
        isFirstVisit={isFirstVisit}
        forceStart={forceStartTour}
        onStartTour={() => {}}
        onSkipTour={() => setForceStartTour(false)}
        onInjectConfig={handleInjectNodeConfig}
        onDeleteNode={handleDeleteNode}
        onDeleteEdge={(edgeId) => setEdges(prev => prev.filter(e => e.id !== edgeId))}
        onCloseConfigDrawer={() => setShowConfigDrawer(false)}
        onOpenTestModal={handleTestClick}
        nodes={nodes}
        edges={edges}
        selectedNodeId={selectedNodeId}
        showConfigDrawer={showConfigDrawer}
        isTestRunning={isTestRunning}
        testResults={testResults}
        onTourComplete={() => setForceStartTour(false)}
        onTourActiveChange={setIsTourActive}
        sidebarActiveTab={sidebarActiveTab}
        showTestModal={showTestConfig}
      />
    </div>
  );
};

export default VisualFlowEditor;
