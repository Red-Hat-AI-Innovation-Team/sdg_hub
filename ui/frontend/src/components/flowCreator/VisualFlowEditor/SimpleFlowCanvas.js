import React, { useState } from 'react';
import { Spinner } from '@patternfly/react-core';
import { CheckCircleIcon, ExclamationCircleIcon } from '@patternfly/react-icons';
import { NODE_TYPE_CONFIG } from './constants';

/**
 * Get the ACTUAL output column name(s) a node will produce.
 * This is the real column name after serialization, not the user's config value.
 */
const getActualOutputColumnName = (node) => {
  const config = node.config || {};
  if (node.type === 'llm') {
    const ext = config._extractor_block_config;
    if (ext?.field_prefix) return `${ext.field_prefix}content`;
    if (ext?.block_name) return `${ext.block_name}_content`;
    return `extract_${config.block_name || 'output'}_content`;
  }
  if (node.type === 'parser') {
    const cols = Array.isArray(config.output_cols) ? config.output_cols : [config.output_cols];
    return cols.filter(Boolean).join(', ') || 'output';
  }
  if (node.type === 'eval') {
    return `${config.block_name || 'eval'}_judgment`;
  }
  if (config.output_cols) {
    const cols = Array.isArray(config.output_cols) ? config.output_cols : [config.output_cols];
    return cols.filter(Boolean).join(', ');
  }
  return 'output';
};

/**
 * Get preview text for a node based on its configuration.
 * Shows the ACTUAL produced column names so users know what to use downstream.
 */
const getNodePreview = (node) => {
  const config = node.config || {};
  
  switch (node.type) {
    case 'llm':
      const inputVars = Array.isArray(config.input_cols) 
        ? config.input_cols.slice(0, 2).join(', ') 
        : '';
      const llmOutput = getActualOutputColumnName(node);
      return inputVars 
        ? `{{${inputVars}}} → ${llmOutput}`
        : `→ ${llmOutput}`;
    case 'parser':
      const parserOutput = getActualOutputColumnName(node);
      return `→ ${parserOutput}`;
    case 'eval':
      return `Pass: ${config.filter_value || 'YES'}`;
    case 'transform':
      return config.transform_type || 'Transform';
    default:
      return 'Configured';
  }
};

/**
 * Simple Flow Canvas Component
 * Renders nodes as draggable divs with SVG edges
 */
const SimpleFlowCanvas = React.memo(({
  nodes,
  edges,
  selectedNodeId,
  onNodeSelect,
  onNodePositionChange,
  onAddEdge,
  onDeleteNode,
  onDeleteEdge,
  // Test execution props
  isTestRunning = false,
  testResults = {},
  activeTestNode = null,
  activeTestEdges = [],
  completedTestNodes = [],
  onNodeClickForIO,
}) => {
  const [draggingNodeId, setDraggingNodeId] = useState(null);
  const [dragPosition, setDragPosition] = useState(null); // Local position while dragging
  const [connectingFrom, setConnectingFrom] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredConnector, setHoveredConnector] = useState(null);
  const [edgeRenderKey, setEdgeRenderKey] = useState(0); // Force SVG re-render
  const [selectedEdgeId, setSelectedEdgeId] = useState(null); // Selected edge for deletion
  const canvasRef = React.useRef(null);
  const dragStartRef = React.useRef({ mouseX: 0, mouseY: 0, nodeX: 0, nodeY: 0 });
  const prevEdgesRef = React.useRef([]);

  // Keyboard handler for deleting selected edges
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedEdgeId) {
        const activeElement = document.activeElement;
        const isInputField = activeElement && (
          activeElement.tagName === 'INPUT' ||
          activeElement.tagName === 'TEXTAREA' ||
          activeElement.isContentEditable
        );
        if (!isInputField) {
          e.preventDefault();
          if (onDeleteEdge) onDeleteEdge(selectedEdgeId);
          setSelectedEdgeId(null);
        }
      }
      if (e.key === 'Escape') {
        setSelectedEdgeId(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedEdgeId, onDeleteEdge]);

  // Pre-compute which nodes have outgoing/incoming edges for connector visibility
  const nodesWithOutgoing = React.useMemo(() => {
    const set = new Set();
    edges.forEach(e => set.add(e.source));
    return set;
  }, [edges]);
  const nodesWithIncoming = React.useMemo(() => {
    const set = new Set();
    edges.forEach(e => set.add(e.target));
    return set;
  }, [edges]);

  // Force SVG re-render when edges change to ensure they paint correctly
  React.useEffect(() => {
    // Check if edges actually changed (not just reference)
    const edgeIds = edges.map(e => e.id).sort().join(',');
    const prevEdgeIds = prevEdgesRef.current.map(e => e.id).sort().join(',');
    
    if (edgeIds !== prevEdgeIds && edges.length > 0) {
      // Slight delay to ensure DOM has updated
      const timer = setTimeout(() => {
        setEdgeRenderKey(k => k + 1);
      }, 50);
      prevEdgesRef.current = edges;
      return () => clearTimeout(timer);
    }
  }, [edges]);

  /**
   * Get test status class for a node
   */
  const getNodeTestClass = (nodeId) => {
    if (activeTestNode === nodeId) return 'node-test-running';
    if (testResults[nodeId]?.status === 'error') return 'node-test-error';
    if (completedTestNodes.includes(nodeId)) return 'node-test-complete';
    return '';
  };

  /**
   * Get test status class for an edge
   */
  const getEdgeTestClass = (edgeId) => {
    if (activeTestEdges.includes(edgeId)) return 'edge-test-active';
    // Check if both source and target nodes are complete
    const edge = edges.find(e => e.id === edgeId);
    if (edge && completedTestNodes.includes(edge.source) && completedTestNodes.includes(edge.target)) {
      return 'edge-test-complete';
    }
    return '';
  };

  /**
   * Handle node mouse down for dragging
   * Does NOT open details immediately - waits for mouseUp to distinguish click vs drag
   */
  const handleNodeMouseDown = (e, node) => {
    if (e.target.closest('.node-connector')) return;
    
    e.stopPropagation();
    e.preventDefault();
    
    const nodeX = node.position?.x || 0;
    const nodeY = node.position?.y || 0;
    
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      nodeX: nodeX,
      nodeY: nodeY,
      hasDragged: false, // Track if mouse actually moved (drag vs click)
      nodeId: node.id,
    };
    
    setDraggingNodeId(node.id);
    setDragPosition({ x: nodeX, y: nodeY });
  };

  /**
   * Handle mouse move for dragging nodes or drawing connection line
   */
  const handleMouseMove = (e) => {
    // Handle connection line preview
    if (connectingFrom && canvasRef.current) {
      const canvasRect = canvasRef.current.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - canvasRect.left,
        y: e.clientY - canvasRect.top,
      });
    }
    
    // Handle node dragging - update local position only
    if (draggingNodeId) {
      const deltaX = e.clientX - dragStartRef.current.mouseX;
      const deltaY = e.clientY - dragStartRef.current.mouseY;
      
      // Mark as dragged if mouse moved more than 3px (threshold to distinguish click vs drag)
      if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
        dragStartRef.current.hasDragged = true;
      }
      
      setDragPosition({
        x: Math.max(0, dragStartRef.current.nodeX + deltaX),
        y: Math.max(0, dragStartRef.current.nodeY + deltaY),
      });
    }
  };

  /**
   * Handle mouse up to stop dragging or cancel connection
   * Only opens node details if it was a click (no drag movement)
   */
  const nodeClickedRef = React.useRef(false);

  const handleMouseUp = () => {
    if (draggingNodeId) {
      // Commit the final position to parent
      if (dragPosition) {
        onNodePositionChange(draggingNodeId, dragPosition);
      }
      
      // Only open details if user clicked without dragging
      if (!dragStartRef.current.hasDragged) {
        const clickedNodeId = dragStartRef.current.nodeId;
        // Flag that a node was clicked so the canvas onClick doesn't deselect it
        nodeClickedRef.current = true;
        // If there's a test result for this node, open the I/O modal instead of config
        if (testResults[clickedNodeId] && onNodeClickForIO) {
          onNodeClickForIO(clickedNodeId);
        } else {
          onNodeSelect(clickedNodeId);
        }
        setSelectedEdgeId(null); // Deselect any edge
      }
    }
    
    setDraggingNodeId(null);
    setDragPosition(null);
    setConnectingFrom(null);
  };
  
  /**
   * Get the display position for a node (use drag position if currently dragging)
   */
  const getNodeDisplayPosition = (node) => {
    if (draggingNodeId === node.id && dragPosition) {
      return dragPosition;
    }
    return node.position || { x: 0, y: 0 };
  };

  /**
   * Start dragging a connection from a connector
   * Silently prevents starting if the node already has an outgoing edge
   */
  const handleConnectorMouseDown = (e, node, isOutput) => {
    e.stopPropagation();
    e.preventDefault();
    
    if (isOutput) {
      // Don't allow starting a connection if this node already has an outgoing edge
      const hasOutgoing = edges.some(edge => edge.source === node.id);
      if (hasOutgoing) return;

      setConnectingFrom(node.id);
      // Set initial mouse position
      if (canvasRef.current) {
        const canvasRect = canvasRef.current.getBoundingClientRect();
        setMousePosition({
          x: e.clientX - canvasRect.left,
          y: e.clientY - canvasRect.top,
        });
      }
    }
  };

  /**
   * Complete connection on any connector (all connectors can be drop targets)
   * Silently rejects if the target node already has an incoming edge
   */
  const handleConnectorMouseUp = (e, node, isOutput) => {
    e.stopPropagation();
    
    if (connectingFrom && connectingFrom !== node.id) {
      // Only complete if target doesn't already have an incoming edge
      const hasIncoming = edges.some(edge => edge.target === node.id);
      if (!hasIncoming) {
        onAddEdge(connectingFrom, node.id);
      }
    }
    setConnectingFrom(null);
  };

  // Fixed node dimensions (must match CSS in node rendering)
  const NODE_WIDTH = 200;
  const NODE_HEIGHT = 88;

  /**
   * Get node connector position for edge drawing
   * @param {string} nodeId - The node ID
   * @param {string} side - 'left', 'right', 'top', or 'bottom'
   */
  const getConnectorPosition = (nodeId, side) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };
    
    // Use display position (which includes drag position if dragging)
    const pos = getNodeDisplayPosition(node);
    
    switch (side) {
      case 'right':
        return { x: pos.x + NODE_WIDTH, y: pos.y + NODE_HEIGHT / 2 };
      case 'left':
        return { x: pos.x, y: pos.y + NODE_HEIGHT / 2 };
      case 'top':
        return { x: pos.x + NODE_WIDTH / 2, y: pos.y };
      case 'bottom':
        return { x: pos.x + NODE_WIDTH / 2, y: pos.y + NODE_HEIGHT };
      default:
        // Legacy: isOutput boolean support
        return {
          x: pos.x + (side ? NODE_WIDTH : 0),
          y: pos.y + NODE_HEIGHT / 2,
        };
    }
  };

  /**
   * Determine the best connector sides for an edge based on node positions
   * Returns { sourceSide, targetSide } indicating which connectors to use
   */
  const getBestConnectorSides = (sourceNode, targetNode) => {
    const sourcePos = getNodeDisplayPosition(sourceNode);
    const targetPos = getNodeDisplayPosition(targetNode);
    const dx = targetPos.x - sourcePos.x;
    const dy = targetPos.y - sourcePos.y;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    // Vertical transition: nodes are roughly in the same column, different rows
    if (absDy > 20 && absDx < 250) {
      if (dy > 0) {
        return { sourceSide: 'bottom', targetSide: 'top' };
      } else {
        return { sourceSide: 'top', targetSide: 'bottom' };
      }
    }

    // Horizontal: target is to the right of source
    if (dx > 0) {
      return { sourceSide: 'right', targetSide: 'left' };
    }

    // Horizontal: target is to the left of source (reversed row)
    return { sourceSide: 'left', targetSide: 'right' };
  };

  // Dynamically calculate canvas size based on node positions
  // Expands when nodes are near the edges, with a 400px buffer
  const canvasSize = React.useMemo(() => {
    const BUFFER = 400;
    const MIN_WIDTH = 1200;
    const MIN_HEIGHT = 800;
    
    let maxX = 0;
    let maxY = 0;
    
    nodes.forEach(node => {
      const pos = node.position || { x: 0, y: 0 };
      maxX = Math.max(maxX, pos.x + NODE_WIDTH);
      maxY = Math.max(maxY, pos.y + NODE_HEIGHT);
    });
    
    return {
      width: Math.max(MIN_WIDTH, maxX + BUFFER),
      height: Math.max(MIN_HEIGHT, maxY + BUFFER),
    };
  }, [nodes]);

  return (
    <div
      ref={canvasRef}
      style={{
        minWidth: `${canvasSize.width}px`,
        minHeight: `${canvasSize.height}px`,
        position: 'relative',
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={() => {
        // Don't deselect if a node was just clicked (mouseUp already handled it)
        if (nodeClickedRef.current) {
          nodeClickedRef.current = false;
          return;
        }
        onNodeSelect(null);
        setSelectedEdgeId(null);
        setConnectingFrom(null);
      }}
    >
      {/* SVG for edges - key forces re-render when edges change to ensure paint */}
      <svg
        key={`edges-svg-${edgeRenderKey}`}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          minWidth: `${canvasSize.width}px`,
          minHeight: `${canvasSize.height}px`,
          pointerEvents: 'none',
          zIndex: 5,
          overflow: 'visible',
        }}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#0066cc" />
          </marker>
          <marker
            id="arrowhead-preview"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#0066cc" opacity="0.5" />
          </marker>
        </defs>
        
        {/* Render existing edges - only if both source and target nodes exist */}
        {edges.map(edge => {
          // Validate that both source and target nodes exist
          const sourceNode = nodes.find(n => n.id === edge.source);
          const targetNode = nodes.find(n => n.id === edge.target);
          
          // Skip rendering if either node doesn't exist (prevents rendering at 0,0)
          if (!sourceNode || !targetNode) {
            return null;
          }
          
          // Determine the best connector sides based on node positions
          const { sourceSide, targetSide } = getBestConnectorSides(sourceNode, targetNode);
          const start = getConnectorPosition(edge.source, sourceSide);
          const end = getConnectorPosition(edge.target, targetSide);
          
          // Build bezier curve based on connector directions
          let edgePath;
          const absDx = Math.abs(end.x - start.x);
          const absDy = Math.abs(end.y - start.y);
          
          if (sourceSide === 'bottom' && targetSide === 'top') {
            // Vertical down: control points extend vertically
            const controlOffset = Math.max(30, absDy * 0.4);
            edgePath = `M ${start.x} ${start.y} C ${start.x} ${start.y + controlOffset}, ${end.x} ${end.y - controlOffset}, ${end.x} ${end.y}`;
          } else if (sourceSide === 'top' && targetSide === 'bottom') {
            // Vertical up: control points extend vertically upward
            const controlOffset = Math.max(30, absDy * 0.4);
            edgePath = `M ${start.x} ${start.y} C ${start.x} ${start.y - controlOffset}, ${end.x} ${end.y + controlOffset}, ${end.x} ${end.y}`;
          } else if (sourceSide === 'left' && targetSide === 'right') {
            // Right to left: control points extend outward (left from source, right from target)
            const controlOffset = Math.max(50, absDx * 0.4);
            edgePath = `M ${start.x} ${start.y} C ${start.x - controlOffset} ${start.y}, ${end.x + controlOffset} ${end.y}, ${end.x} ${end.y}`;
          } else {
            // Left to right (default): control points extend right from source, left from target
            const controlOffset = Math.max(50, absDx * 0.4);
            edgePath = `M ${start.x} ${start.y} C ${start.x + controlOffset} ${start.y}, ${end.x - controlOffset} ${end.y}, ${end.x} ${end.y}`;
          }
          
          const edgeTestClass = getEdgeTestClass(edge.id);
          
          const isEdgeSelected = selectedEdgeId === edge.id;
          
          return (
            <g key={edge.id}>
              {/* Invisible wider path for easier click target */}
              <path
                d={edgePath}
                fill="none"
                stroke="transparent"
                strokeWidth="14"
                style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedEdgeId(edge.id);
                  onNodeSelect(null); // Deselect any node
                }}
              />
              {/* Visible edge path */}
              <path
                className={edgeTestClass}
                d={edgePath}
                fill="none"
                stroke={isEdgeSelected ? '#c9190b' : edgeTestClass === 'edge-test-complete' ? '#3e8635' : '#0066cc'}
                strokeWidth={isEdgeSelected ? 3 : 2}
                markerEnd="url(#arrowhead)"
                style={{ pointerEvents: 'none' }}
              />
            </g>
          );
        })}

        {/* Render connection preview line while dragging */}
        {connectingFrom && (
          <path
            d={(() => {
              // Find the closest connector side to the mouse position
              const sourceNode = nodes.find(n => n.id === connectingFrom);
              if (!sourceNode) return '';
              const sourcePos = getNodeDisplayPosition(sourceNode);
              const mx = mousePosition.x;
              const my = mousePosition.y;
              const centerX = sourcePos.x + NODE_WIDTH / 2;
              const centerY = sourcePos.y + NODE_HEIGHT / 2;
              const ddx = mx - centerX;
              const ddy = my - centerY;
              
              // Choose the closest side based on mouse direction
              let side = 'right';
              if (Math.abs(ddx) > Math.abs(ddy)) {
                side = ddx > 0 ? 'right' : 'left';
              } else {
                side = ddy > 0 ? 'bottom' : 'top';
              }
              
              const start = getConnectorPosition(connectingFrom, side);
              const end = mousePosition;
              const absDx = Math.abs(end.x - start.x);
              const absDy = Math.abs(end.y - start.y);
              
              if (side === 'bottom' || side === 'top') {
                const controlOffset = Math.max(30, absDy * 0.4);
                const dir = side === 'bottom' ? 1 : -1;
                return `M ${start.x} ${start.y} C ${start.x} ${start.y + controlOffset * dir}, ${end.x} ${end.y - controlOffset * dir}, ${end.x} ${end.y}`;
              } else if (side === 'left') {
                const controlOffset = Math.max(50, absDx * 0.4);
                return `M ${start.x} ${start.y} C ${start.x - controlOffset} ${start.y}, ${end.x + controlOffset} ${end.y}, ${end.x} ${end.y}`;
              } else {
                const controlOffset = Math.max(50, absDx * 0.4);
                return `M ${start.x} ${start.y} C ${start.x + controlOffset} ${start.y}, ${end.x - controlOffset} ${end.y}, ${end.x} ${end.y}`;
              }
            })()}
            fill="none"
            stroke="#0066cc"
            strokeWidth="2"
            strokeDasharray="5,5"
            opacity="0.6"
            markerEnd="url(#arrowhead-preview)"
          />
        )}
      </svg>

      {/* Render nodes */}
      {nodes.map(node => {
        const config = NODE_TYPE_CONFIG[node.type];
        const isSelected = selectedNodeId === node.id;
        const isConnectingSource = connectingFrom === node.id;
        const isValidTarget = connectingFrom && connectingFrom !== node.id;
        const isDragging = draggingNodeId === node.id;
        const displayPos = getNodeDisplayPosition(node);
        const nodeTestClass = getNodeTestClass(node.id);
        const hasTestResult = testResults[node.id];
        const isTestComplete = completedTestNodes.includes(node.id);
        const isTestError = testResults[node.id]?.status === 'error';
        
        // Connector visibility: hide if the node already has the corresponding edge
        const hasOutgoing = nodesWithOutgoing.has(node.id);
        const hasIncoming = nodesWithIncoming.has(node.id);
        // Show connectors for dragging FROM this node (only if no outgoing edge)
        const showAsDragSource = !hasOutgoing;
        // Show connectors as drop targets (only if someone is dragging AND this node has no incoming edge)
        const showAsDropTarget = connectingFrom && connectingFrom !== node.id && !hasIncoming;
        // Show connectors if either condition is met
        const showConnectors = showAsDragSource || showAsDropTarget;

        return (
          <div
            key={node.id}
            data-node-id={node.id}
            className={nodeTestClass}
            style={{
              position: 'absolute',
              left: displayPos.x,
              top: displayPos.y,
              width: '200px',
              height: '88px',
              cursor: hasTestResult ? 'pointer' : isDragging ? 'grabbing' : 'grab',
              userSelect: 'none',
              zIndex: isSelected ? 10 : 1,
            }}
            onMouseDown={(e) => handleNodeMouseDown(e, node)}
          >
            {/* Inner content wrapper - fixed height with overflow hidden */}
            <div style={{
              width: '100%',
              height: '100%',
              overflow: 'hidden',
              background: '#fff',
              border: `2px solid ${
                isTestError ? '#c9190b' :
                isTestComplete ? '#3e8635' :
                activeTestNode === node.id ? '#0066cc' :
                isSelected ? config?.color : 
                isValidTarget ? '#0066cc' : '#d2d2d2'
              }`,
              borderRadius: '8px',
              boxShadow: isSelected 
                ? `0 0 0 2px ${config?.color}40, 0 4px 12px rgba(0,0,0,0.15)` 
                : isValidTarget
                ? '0 0 0 2px rgba(0, 102, 204, 0.3), 0 4px 12px rgba(0,0,0,0.15)'
                : '0 2px 4px rgba(0,0,0,0.1)',
              transition: 'border-color 0.15s, box-shadow 0.15s',
              boxSizing: 'border-box',
            }}>
              {/* Node Header */}
              <div style={{
                padding: '8px 12px',
                borderBottom: '1px solid #d2d2d2',
                background: isTestComplete ? '#e7f5e1' : isTestError ? '#fce8e8' : activeTestNode === node.id ? '#e6f0ff' : `${config?.color}10`,
                borderRadius: '6px 6px 0 0',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                {config?.icon && React.createElement(config.icon, { 
                  style: { color: config?.color, fontSize: '16px' } 
                })}
                <span style={{ 
                  fontWeight: 600, 
                  fontSize: '13px',
                  flex: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {node.config?.block_name || config?.label}
                </span>
                {/* Test status indicators */}
                {isTestComplete && (
                  <CheckCircleIcon style={{ color: '#3e8635', fontSize: '16px' }} title="Test passed - click to view output" />
                )}
                {isTestError && (
                  <ExclamationCircleIcon style={{ color: '#c9190b', fontSize: '16px' }} title="Test failed - click to view error" />
                )}
                {activeTestNode === node.id && (
                  <Spinner size="sm" style={{ width: '16px', height: '16px' }} />
                )}
                {!node.configured && !isTestComplete && !isTestError && activeTestNode !== node.id && (
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: '#f0ab00',
                  }} title="Not configured" />
                )}
              </div>

              {/* Node Body */}
              <div style={{
                padding: '8px 12px',
                fontSize: '12px',
                color: '#6a6e73',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
              }}>
                {node.configured 
                  ? getNodePreview(node)
                  : 'Click to configure'
                }
              </div>
            </div>

            {/* Connectors - only shown when the node can make/receive connections */}
            {showConnectors && (<>
            {/* Input Connector (left) */}
            <div
              className="node-connector node-connector-input"
              style={{
                position: 'absolute',
                left: '-10px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: showAsDropTarget 
                  ? (hoveredConnector === `${node.id}-input` ? '#0066cc' : '#e6f0ff')
                  : '#fff',
                border: `3px solid ${showAsDropTarget ? '#0066cc' : '#8a8d90'}`,
                cursor: showAsDropTarget ? 'pointer' : 'crosshair',
                zIndex: 20,
                transition: 'all 0.15s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onMouseDown={(e) => handleConnectorMouseDown(e, node, true)}
              onMouseUp={(e) => handleConnectorMouseUp(e, node, false)}
              onMouseEnter={() => setHoveredConnector(`${node.id}-input`)}
              onMouseLeave={() => setHoveredConnector(null)}
              title={showAsDropTarget ? "Drop here to connect" : "Drag to connect"}
            >
              {(showAsDropTarget || hoveredConnector === `${node.id}-input`) && (
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: (showAsDropTarget && hoveredConnector === `${node.id}-input`) ? '#fff' : '#0066cc',
                }} />
              )}
            </div>

            {/* Output Connector (right) */}
            <div
              className="node-connector node-connector-output"
              data-tour={node.type === 'llm' ? 'llm-output-port' : undefined}
              style={{
                position: 'absolute',
                right: '-10px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: isConnectingSource ? '#0066cc' : showAsDropTarget 
                  ? (hoveredConnector === `${node.id}-output` ? '#0066cc' : '#e6f0ff')
                  : '#fff',
                border: `3px solid ${isConnectingSource ? '#0066cc' : showAsDropTarget ? '#0066cc' : '#8a8d90'}`,
                cursor: showAsDropTarget ? 'pointer' : 'crosshair',
                zIndex: 20,
                transition: 'all 0.15s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onMouseDown={(e) => handleConnectorMouseDown(e, node, true)}
              onMouseUp={(e) => handleConnectorMouseUp(e, node, false)}
              onMouseEnter={() => setHoveredConnector(`${node.id}-output`)}
              onMouseLeave={() => setHoveredConnector(null)}
              title={showAsDropTarget ? "Drop here to connect" : "Drag to connect"}
            >
              {((hoveredConnector === `${node.id}-output` && !isConnectingSource) || (showAsDropTarget && hoveredConnector === `${node.id}-output`)) && (
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: (showAsDropTarget && hoveredConnector === `${node.id}-output`) ? '#fff' : '#0066cc',
                }} />
              )}
            </div>

            {/* Top Connector */}
            <div
              className="node-connector node-connector-top"
              style={{
                position: 'absolute',
                top: '-10px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: showAsDropTarget 
                  ? (hoveredConnector === `${node.id}-top` ? '#0066cc' : '#e6f0ff')
                  : '#fff',
                border: `3px solid ${showAsDropTarget ? '#0066cc' : '#8a8d90'}`,
                cursor: showAsDropTarget ? 'pointer' : 'crosshair',
                zIndex: 20,
                transition: 'all 0.15s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onMouseDown={(e) => handleConnectorMouseDown(e, node, true)}
              onMouseUp={(e) => handleConnectorMouseUp(e, node, false)}
              onMouseEnter={() => setHoveredConnector(`${node.id}-top`)}
              onMouseLeave={() => setHoveredConnector(null)}
              title={showAsDropTarget ? "Drop here to connect" : "Drag to connect"}
            >
              {(showAsDropTarget || hoveredConnector === `${node.id}-top`) && (
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: (showAsDropTarget && hoveredConnector === `${node.id}-top`) ? '#fff' : '#0066cc',
                }} />
              )}
            </div>

            {/* Bottom Connector */}
            <div
              className="node-connector node-connector-bottom"
              style={{
                position: 'absolute',
                bottom: '-10px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: showAsDropTarget 
                  ? (hoveredConnector === `${node.id}-bottom` ? '#0066cc' : '#e6f0ff')
                  : '#fff',
                border: `3px solid ${showAsDropTarget ? '#0066cc' : '#8a8d90'}`,
                cursor: showAsDropTarget ? 'pointer' : 'crosshair',
                zIndex: 20,
                transition: 'all 0.15s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onMouseDown={(e) => handleConnectorMouseDown(e, node, true)}
              onMouseUp={(e) => handleConnectorMouseUp(e, node, false)}
              onMouseEnter={() => setHoveredConnector(`${node.id}-bottom`)}
              onMouseLeave={() => setHoveredConnector(null)}
              title={showAsDropTarget ? "Drop here to connect" : "Drag to connect"}
            >
              {(showAsDropTarget || hoveredConnector === `${node.id}-bottom`) && (
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: (showAsDropTarget && hoveredConnector === `${node.id}-bottom`) ? '#fff' : '#0066cc',
                }} />
              )}
            </div>
            </>)}
          </div>
        );
      })}
    </div>
  );
});

SimpleFlowCanvas.displayName = 'SimpleFlowCanvas';

export default SimpleFlowCanvas;
