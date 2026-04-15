import React, { useState, useEffect } from 'react';
import {
  Title,
  SearchInput,
  Badge,
  Tabs,
  Tab,
  TabTitleText,
  Spinner,
  Button,
  Tooltip,
} from '@patternfly/react-core';
import { 
  InfoCircleIcon, 
  CatalogIcon, 
  BookOpenIcon,
  GlobeIcon,
  BullseyeIcon,
  SearchIcon,
  ChartBarIcon,
  PaintBrushIcon,
} from '@patternfly/react-icons';

import { NODE_TYPE_CONFIG, NODE_TYPES, generateNodeId } from './constants';
import { NODE_CONFIGS } from './nodes';

/**
 * Flow template category definitions
 * Maps category IDs to display names and descriptions
 * Matches the structure in FlowSelectionStep.js for consistency
 */
const FLOW_CATEGORIES = {
  'knowledge-generation': {
    name: 'Knowledge Generation',
    description: 'Generate Q&A pairs and knowledge datasets from documents',
    icon: BookOpenIcon,
    color: '#3e8635',
    order: 1,
  },
  'multilingual': {
    name: 'Multilingual',
    description: 'Generate datasets in multiple languages',
    icon: GlobeIcon,
    color: '#8a8d90',
    order: 2,
  },
  'skills-generation': {
    name: 'Skills Generation',
    description: 'Create skill-based training datasets',
    icon: BullseyeIcon,
    color: '#f0ab00',
    order: 3,
  },
  'text-analysis': {
    name: 'Text Analysis',
    description: 'Extract insights and analyze text content',
    icon: SearchIcon,
    color: '#6753ac',
    order: 4,
  },
  'evaluation': {
    name: 'Evaluation',
    description: 'Evaluate and benchmark RAG systems',
    icon: ChartBarIcon,
    color: '#009596',
    order: 5,
  },
  'custom': {
    name: 'Custom Flows',
    description: 'User-created custom flows',
    icon: PaintBrushIcon,
    color: '#0066cc',
    order: 99,
  },
};

/**
 * Categorize a flow template based on its name and tags
 * Order matters! More specific checks come first.
 * Name-based checks take priority for disambiguation.
 * Matches the logic in FlowSelectionStep.js for consistency.
 */
const categorizeFlow = (flowName, tags = []) => {
  const nameLower = flowName.toLowerCase();
  
  // Check if it's a custom flow
  if (flowName.includes('(Custom)')) {
    return 'custom';
  }
  
  // NAME-BASED CHECKS FIRST (more reliable for disambiguation)
  // Check evaluation FIRST - "RAG Evaluation" should go to evaluation, not knowledge
  if (nameLower.includes('evaluation') || nameLower.includes('benchmark')) {
    return 'evaluation';
  }
  // Check multilingual/language-specific BEFORE knowledge - "Japanese...Knowledge Tuning" should go to multilingual
  if (nameLower.includes('japanese') || nameLower.includes('multilingual') || nameLower.includes('korean') || nameLower.includes('chinese') || nameLower.includes('spanish') || nameLower.includes('french') || nameLower.includes('german')) {
    return 'multilingual';
  }
  
  // TAG-BASED CHECKS (for flows without strong name indicators)
  if (tags.includes('evaluation') || tags.includes('benchmark')) {
    return 'evaluation';
  }
  if (tags.includes('multilingual') || tags.includes('translation') || tags.includes('non-english')) {
    return 'multilingual';
  }
  if (tags.includes('skills') || tags.includes('instructlab')) {
    return 'skills-generation';
  }
  if (tags.includes('text-analysis') || tags.includes('extraction') || tags.includes('insights')) {
    return 'text-analysis';
  }
  // rag tag goes to evaluation since most RAG-tagged flows are evaluation-related
  if (tags.includes('rag')) {
    return 'evaluation';
  }
  // qa-pairs and knowledge-tuning are checked LAST as they are generic
  if (tags.includes('knowledge-tuning') || tags.includes('question-generation') || tags.includes('qa-pairs')) {
    return 'knowledge-generation';
  }
  
  // REMAINING NAME-BASED FALLBACKS
  if (nameLower.includes('skill') || nameLower.includes('instructlab')) {
    return 'skills-generation';
  }
  if (nameLower.includes('analysis') || nameLower.includes('extract') || nameLower.includes('insight')) {
    return 'text-analysis';
  }
  // Knowledge generation is the catch-all for Q&A and document processing flows
  if (nameLower.includes('qa') || nameLower.includes('knowledge') || nameLower.includes('summary') || nameLower.includes('rag')) {
    return 'knowledge-generation';
  }
  
  return 'knowledge-generation'; // Default category
};

/**
 * Node Sidebar Component
 * 
 * Displays a palette of available node types and flow templates.
 * Two tabs: "Node Library" for individual nodes, "Flow Templates" for complete flows.
 */
const NodeSidebar = ({ onAddNode, onDragStart, onDragEnd, onLoadFlowTemplate, onTabChange }) => {
  const [activeTab, setActiveTab] = useState(0);
  
  // Notify parent when tab changes
  const handleTabChange = (event, tabIndex) => {
    setActiveTab(tabIndex);
    onTabChange?.(tabIndex === 0 ? 'node-library' : 'flow-templates');
  };
  const [searchValue, setSearchValue] = useState('');
  const [draggingType, setDraggingType] = useState(null);
  const [flowTemplates, setFlowTemplates] = useState([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [templatesError, setTemplatesError] = useState(null);

  /**
   * Node categories for organization
   */
  const NODE_CATEGORIES = {
    generation: {
      label: 'Generation',
      description: 'LLM-based content generation',
      nodes: [
        NODE_TYPES.LLM,
        NODE_TYPES.PARSER,
      ],
    },
    evaluation: {
      label: 'Evaluation',
      description: 'Quality evaluation and filtering',
      nodes: [
        NODE_TYPES.EVAL,
      ],
    },
    data: {
      label: 'Data Transform',
      description: 'Column manipulation operations',
      nodes: [
        NODE_TYPES.TRANSFORM,
      ],
    },
  };

  /**
   * Fetch flow templates when templates tab is selected
   */
  useEffect(() => {
    if (activeTab === 1 && flowTemplates.length === 0 && !loadingTemplates) {
      fetchFlowTemplates();
    }
  }, [activeTab]);

  /**
   * Fetch flow templates from API
   */
  const fetchFlowTemplates = async () => {
    setLoadingTemplates(true);
    setTemplatesError(null);
    try {
      const response = await fetch('/api/flows/templates');
      if (!response.ok) throw new Error('Failed to fetch templates');
      const data = await response.json();
      setFlowTemplates(data.templates || []);
    } catch (error) {
      console.error('Error fetching flow templates:', error);
      setTemplatesError(error.message);
    } finally {
      setLoadingTemplates(false);
    }
  };

  /**
   * Filter nodes by search value
   */
  const filterNodes = (nodeTypes) => {
    if (!searchValue.trim()) return nodeTypes;
    
    const searchLower = searchValue.toLowerCase();
    return nodeTypes.filter(type => {
      const config = NODE_TYPE_CONFIG[type];
      return (
        config.label.toLowerCase().includes(searchLower) ||
        config.description.toLowerCase().includes(searchLower)
      );
    });
  };

  /**
   * Filter templates by search value
   */
  const filterTemplates = (templates) => {
    if (!searchValue.trim()) return templates;
    
    const searchLower = searchValue.toLowerCase();
    return templates.filter(template => 
      template.name.toLowerCase().includes(searchLower) ||
      (template.description || '').toLowerCase().includes(searchLower) ||
      (template.tags || []).some(tag => tag.toLowerCase().includes(searchLower))
    );
  };

  /**
   * Handle drag start - creates a drag preview that matches the node card appearance
   */
  const handleDragStart = (e, nodeType) => {
    setDraggingType(nodeType);
    e.dataTransfer.setData('nodeType', nodeType);
    e.dataTransfer.effectAllowed = 'copy';
    
    const config = NODE_TYPE_CONFIG[nodeType];
    const dragImage = document.createElement('div');
    dragImage.className = 'node-drag-preview';
    
    // Create a drag preview that matches the node card style with SVG icon
    dragImage.innerHTML = `
      <div style="
        display: flex;
        align-items: center;
        padding: 10px 12px;
        background: #fff;
        border: 1px solid ${config.color};
        border-left: 4px solid ${config.color};
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        min-width: 160px;
        max-width: 200px;
      ">
        <div style="
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: ${config.color}15;
          border-radius: 6px;
          margin-right: 10px;
          color: ${config.color};
        ">
          ${config.iconSvg || ''}
        </div>
        <div style="flex: 1; min-width: 0;">
          <div style="
            font-weight: 600;
            font-size: 13px;
            color: #151515;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          ">${config.label}</div>
          <div style="
            font-size: 11px;
            color: #6a6e73;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          ">${config.shortDescription || ''}</div>
        </div>
      </div>
    `;
    
    dragImage.style.cssText = `
      position: absolute;
      top: -1000px;
      left: -1000px;
      z-index: 9999;
    `;
    
    document.body.appendChild(dragImage);
    e.dataTransfer.setDragImage(dragImage, 90, 25);
    
    setTimeout(() => {
      document.body.removeChild(dragImage);
    }, 0);

    if (onDragStart) {
      onDragStart(nodeType);
    }
  };

  /**
   * Handle drag end
   */
  const handleDragEnd = (e) => {
    setDraggingType(null);
    if (onDragEnd) {
      onDragEnd();
    }
  };

  /**
   * Handle double-click to add node
   */
  const handleDoubleClick = (nodeType) => {
    const position = {
      x: 300 + Math.random() * 100,
      y: 150 + Math.random() * 100,
    };
    onAddNode(nodeType, position);
  };

  /**
   * Fetch prompt content from a prompt file path
   */
  const fetchPromptContent = async (promptPath) => {
    try {
      const response = await fetch(`/api/prompts/load?prompt_path=${encodeURIComponent(promptPath)}`);
      if (!response.ok) {
        console.warn(`Failed to load prompt: ${promptPath}`);
        return null;
      }
      const data = await response.json();
      return data.messages || [];
    } catch (error) {
      console.warn(`Error loading prompt ${promptPath}:`, error);
      return null;
    }
  };

  /**
   * Extract system and user messages from prompt messages array
   */
  const extractPromptMessages = (messages) => {
    if (!messages || !Array.isArray(messages)) {
      return { system_message: '', user_message: '' };
    }
    
    const systemMsg = messages.find(m => m.role === 'system');
    const userMsg = messages.find(m => m.role === 'user');
    
    return {
      system_message: systemMsg?.content || '',
      user_message: userMsg?.content || '',
    };
  };

  /**
   * Convert flow blocks to visual nodes
   */
  const convertBlocksToVisualNodes = (blocks, promptsMap = {}) => {
    const visualNodes = [];
    const visualEdges = [];
    const promptPathsToFetch = []; // Track which prompts need fetching
    let nodeIndex = 0;
    let currentX = 50;
    let currentY = 50;
    const nodeSpacingX = 280;
    const nodeSpacingY = 140;
    const nodesPerRow = 3;

    // Group blocks into logical visual nodes
    let i = 0;
    while (i < blocks.length) {
      const block = blocks[i];
      const blockType = block.block_type;
      const blockConfig = block.block_config || {};

      // Calculate position in serpentine (snake) layout
      // Even rows (0, 2, 4...): left to right
      // Odd rows (1, 3, 5...): right to left
      const row = Math.floor(nodeIndex / nodesPerRow);
      const col = nodeIndex % nodesPerRow;
      const isReversedRow = row % 2 === 1;
      const adjustedCol = isReversedRow ? (nodesPerRow - 1 - col) : col;
      const position = {
        x: currentX + adjustedCol * nodeSpacingX,
        y: currentY + row * nodeSpacingY,
      };

      // Check for LLM pattern: PromptBuilder + LLMChat + LLMResponseExtractor
      if (blockType === 'PromptBuilderBlock') {
        // Use full_prompt_path if available, otherwise fall back to prompt_config_path
        const promptPath = blockConfig.full_prompt_path || blockConfig.prompt_config_path;
        const promptMessages = promptsMap[promptPath] || null;
        const { system_message, user_message } = extractPromptMessages(promptMessages);
        
        // Look ahead to see if this is part of an LLM or Eval pattern
        const nextBlock = blocks[i + 1];
        const thirdBlock = blocks[i + 2];
        
        if (nextBlock?.block_type === 'LLMChatBlock' && 
            thirdBlock?.block_type === 'LLMResponseExtractorBlock') {
          
          // Check if this is an Eval pattern (has Parser + Filter after)
          // Pattern: PromptBuilder + LLMChat + Extractor + Parser + Filter
          const fourthBlock = blocks[i + 3];
          const fifthBlock = blocks[i + 4];
          
          if ((fourthBlock?.block_type === 'TagParserBlock' || fourthBlock?.block_type === 'RegexParserBlock') &&
              fifthBlock?.block_type === 'ColumnValueFilterBlock') {
            // This is an Eval node (any prompt+llm+extractor+parser+filter pattern)
            const nodeId = generateNodeId();
            // Preserve input_cols format (array, dict, or string)
            const evalInputCols = Array.isArray(blockConfig.input_cols)
              ? blockConfig.input_cols
              : (typeof blockConfig.input_cols === 'object' && blockConfig.input_cols !== null)
                ? blockConfig.input_cols
                : [blockConfig.input_cols].filter(Boolean);
            visualNodes.push({
              id: nodeId,
              type: NODE_TYPES.EVAL,
              label: blockConfig.block_name || 'Eval',
              position,
              configured: true,
              config: {
                block_name: blockConfig.block_name,
                input_cols: evalInputCols,
                output_cols: fourthBlock.block_config?.output_cols || [],
                system_message,
                user_message,
                prompt_config_path: promptPath,
                start_tags: fourthBlock?.block_type === 'TagParserBlock'
                  ? (fourthBlock.block_config?.start_tags || [])
                  : [],
                end_tags: fourthBlock?.block_type === 'TagParserBlock'
                  ? (fourthBlock.block_config?.end_tags || [])
                  : [],
                parsing_pattern: fourthBlock?.block_type === 'RegexParserBlock'
                  ? (fourthBlock.block_config?.parsing_pattern || '')
                  : '',
                filter_value: fifthBlock.block_config?.filter_value || 'YES',
                filter_operation: fifthBlock.block_config?.operation || 'eq',
                max_tokens: nextBlock.block_config?.max_tokens || 2048,
                temperature: nextBlock.block_config?.temperature || 0.7,
              },
            });
            
            if (promptPath && !promptMessages) {
              promptPathsToFetch.push({ nodeId, promptPath });
            }
            
            i += 5; // Skip all 5 blocks
            nodeIndex++;
            continue;
          }
          
          // This is a regular LLM node
          const nodeId = generateNodeId();
          const extractorConfig = thirdBlock.block_config || {};
          // Preserve input_cols format: arrays stay as arrays, dicts stay as dicts,
          // strings get wrapped in an array. PromptBuilderBlock accepts all three.
          const promptInputCols = Array.isArray(blockConfig.input_cols)
            ? blockConfig.input_cols
            : (typeof blockConfig.input_cols === 'object' && blockConfig.input_cols !== null)
              ? blockConfig.input_cols  // keep dict as-is (e.g. {document: 'document', topic: 'topic'})
              : [blockConfig.input_cols].filter(Boolean);
          visualNodes.push({
            id: nodeId,
            type: NODE_TYPES.LLM,
            label: nextBlock.block_config?.block_name || blockConfig.block_name || 'LLM',
            position,
            configured: true,
            config: {
              block_name: nextBlock.block_config?.block_name || blockConfig.block_name,
              input_cols: promptInputCols,
              output_cols: blockConfig.output_cols || '',
              system_message,
              user_message,
              prompt_config_path: promptPath,
              max_tokens: nextBlock.block_config?.max_tokens || 2048,
              temperature: nextBlock.block_config?.temperature || 0.7,
              n: nextBlock.block_config?.n || 1,
              async_mode: nextBlock.block_config?.async_mode ?? true,
              // Preserve original LLM block config for correct column name round-tripping
              _llm_block_config: {
                block_name: nextBlock.block_config?.block_name,
                input_cols: nextBlock.block_config?.input_cols,
                output_cols: nextBlock.block_config?.output_cols,
                max_tokens: nextBlock.block_config?.max_tokens,
                temperature: nextBlock.block_config?.temperature,
                n: nextBlock.block_config?.n,
                async_mode: nextBlock.block_config?.async_mode,
              },
              // Preserve extractor config including field_prefix for correct
              // output column naming (e.g. field_prefix: "topic_" → "topic_content")
              _extractor_block_config: {
                block_name: extractorConfig.block_name,
                input_cols: extractorConfig.input_cols,
                output_cols: extractorConfig.output_cols,
                field_prefix: extractorConfig.field_prefix,
                extract_content: extractorConfig.extract_content,
                extract_reasoning_content: extractorConfig.extract_reasoning_content,
                expand_lists: extractorConfig.expand_lists,
              },
            },
          });
          
          if (promptPath && !promptMessages) {
            promptPathsToFetch.push({ nodeId, promptPath });
          }
          
          i += 3; // Skip all 3 blocks
          nodeIndex++;
          continue;
        }
      }

      // TagParserBlock or RegexParserBlock -> Parser node
      if (blockType === 'TagParserBlock' || blockType === 'RegexParserBlock') {
        visualNodes.push({
          id: generateNodeId(),
          type: NODE_TYPES.PARSER,
          label: blockConfig.block_name || 'Parser',
          position,
          configured: true,
          config: {
            block_name: blockConfig.block_name,
            input_cols: blockConfig.input_cols || '',
            output_cols: Array.isArray(blockConfig.output_cols) ? blockConfig.output_cols : [blockConfig.output_cols].filter(Boolean),
            start_tags: blockConfig.start_tags || [],
            end_tags: blockConfig.end_tags || [],
            parsing_pattern: blockConfig.parsing_pattern || '',
          },
        });
        i++;
        nodeIndex++;
        continue;
      }

      // Transform blocks
      if (['DuplicateColumnsBlock', 'RenameColumnsBlock', 'MeltColumnsBlock'].includes(blockType)) {
        let transformType = 'duplicate';
        if (blockType === 'RenameColumnsBlock') transformType = 'rename';
        if (blockType === 'MeltColumnsBlock') transformType = 'melt';

        visualNodes.push({
          id: generateNodeId(),
          type: NODE_TYPES.TRANSFORM,
          label: blockConfig.block_name || 'Transform',
          position,
          configured: true,
          config: {
            block_name: blockConfig.block_name,
            transform_type: transformType,
            input_cols: blockConfig.input_cols || {},
            output_cols: Array.isArray(blockConfig.output_cols) ? blockConfig.output_cols : [blockConfig.output_cols].filter(Boolean),
          },
        });
        i++;
        nodeIndex++;
        continue;
      }

      // Skip other blocks (like LLMResponseExtractor when not part of pattern)
      i++;
    }

    // Create edges connecting nodes in sequence
    // Use unique edge IDs based on node IDs to avoid React Flow rendering issues
    for (let j = 0; j < visualNodes.length - 1; j++) {
      visualEdges.push({
        id: `edge_${visualNodes[j].id}_to_${visualNodes[j + 1].id}`,
        source: visualNodes[j].id,
        target: visualNodes[j + 1].id,
        type: 'default',
      });
    }

    return { visualNodes, visualEdges, promptPathsToFetch };
  };

  /**
   * Handle loading a flow template
   */
  const handleLoadTemplate = async (template) => {
    if (!onLoadFlowTemplate) return;
    
    // First pass: convert blocks without prompts
    const { visualNodes, visualEdges, promptPathsToFetch } = convertBlocksToVisualNodes(template.blocks);
    
    // Fetch prompts BEFORE loading (so we have everything ready)
    if (promptPathsToFetch.length > 0) {
      // Collect unique prompt paths
      const uniquePaths = [...new Set(promptPathsToFetch.map(p => p.promptPath))];
      
      // Fetch all prompts in parallel
      const promptsMap = {};
      await Promise.all(
        uniquePaths.map(async (path) => {
          const messages = await fetchPromptContent(path);
          if (messages) {
            promptsMap[path] = messages;
          }
        })
      );
      
      // Update the visual nodes with fetched prompts (create new objects to ensure React detects changes)
      if (Object.keys(promptsMap).length > 0) {
        for (let i = 0; i < visualNodes.length; i++) {
          const node = visualNodes[i];
          const promptPath = node.config?.prompt_config_path;
          if (promptPath && promptsMap[promptPath]) {
            const { system_message, user_message } = extractPromptMessages(promptsMap[promptPath]);
            // Create new node object with updated config
            visualNodes[i] = {
              ...node,
              config: {
                ...node.config,
                system_message,
                user_message,
              }
            };
          }
        }
      }
    }
    
    // Load the template with prompts already populated
    onLoadFlowTemplate({
      name: template.name,
      metadata: template.metadata,
      visualNodes,
      visualEdges,
      originalBlocks: template.blocks,
      sourceFlowName: template.name, // Track source for prompt file copying
    });
  };

  /**
   * Render a single node item in the palette
   */
  const renderNodeItem = (nodeType) => {
    const config = NODE_TYPE_CONFIG[nodeType];
    const isDragging = draggingType === nodeType;

    return (
      <div
        key={nodeType}
        data-tour={`node-library-${nodeType}`}
        draggable
        onDragStart={(e) => handleDragStart(e, nodeType)}
        onDragEnd={handleDragEnd}
        onDoubleClick={() => handleDoubleClick(nodeType)}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '10px 12px',
          marginBottom: '8px',
          background: isDragging ? `${config.color}20` : '#fff',
          border: `1px solid ${isDragging ? config.color : '#d2d2d2'}`,
          borderLeft: `4px solid ${config.color}`,
          borderRadius: '4px',
          cursor: 'grab',
          transition: 'all 0.15s ease',
          opacity: isDragging ? 0.6 : 1,
          userSelect: 'none',
        }}
        onMouseEnter={(e) => {
          if (!isDragging) {
            e.currentTarget.style.background = '#f0f0f0';
            e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isDragging) {
            e.currentTarget.style.background = '#fff';
            e.currentTarget.style.boxShadow = 'none';
          }
        }}
      >
        {/* Drag Handle */}
        <div style={{ 
          marginRight: '8px', 
          color: '#6a6e73',
          display: 'flex',
          alignItems: 'center',
        }}>
          <svg width="12" height="16" viewBox="0 0 12 16" fill="currentColor">
            <circle cx="3" cy="3" r="1.5"/>
            <circle cx="9" cy="3" r="1.5"/>
            <circle cx="3" cy="8" r="1.5"/>
            <circle cx="9" cy="8" r="1.5"/>
            <circle cx="3" cy="13" r="1.5"/>
            <circle cx="9" cy="13" r="1.5"/>
          </svg>
        </div>

        {/* Node Icon */}
        <div
          style={{
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `${config.color}15`,
            borderRadius: '6px',
            marginRight: '12px',
          }}
        >
          {config.icon && React.createElement(config.icon, { 
            style: { color: config.color, fontSize: '18px' } 
          })}
        </div>

        {/* Node Info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ 
            fontWeight: 600, 
            fontSize: '14px',
            color: '#151515',
            marginBottom: '2px',
          }}>
            {config.label}
          </div>
          <div style={{ 
            fontSize: '12px', 
            color: '#6a6e73',
            lineHeight: '1.4',
          }}>
            {config.description}
          </div>
        </div>
      </div>
    );
  };

  /**
   * Render a flow template item (matching Node Library styling)
   * @param {boolean} isFirst - Whether this is the first template overall (for tour)
   */
  const renderTemplateItem = (template, categoryColor, CategoryIcon, isFirst = false) => {
    return (
      <div
        key={template.id || template.name}
        data-tour={isFirst ? 'first-template' : undefined}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '10px 12px',
          marginBottom: '8px',
          background: '#fff',
          border: '1px solid #d2d2d2',
          borderLeft: `4px solid ${categoryColor}`,
          borderRadius: '4px',
          cursor: 'pointer',
          transition: 'all 0.15s ease',
          userSelect: 'none',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = '#f0f0f0';
          e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = '#fff';
          e.currentTarget.style.boxShadow = 'none';
        }}
        onClick={() => handleLoadTemplate(template)}
      >
        {/* Template Icon */}
        <div
          style={{
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `${categoryColor}15`,
            borderRadius: '6px',
            marginRight: '12px',
            flexShrink: 0,
          }}
        >
          {CategoryIcon && <CategoryIcon style={{ color: categoryColor, fontSize: '18px' }} />}
        </div>

        {/* Template Info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ 
            fontWeight: 600, 
            fontSize: '14px',
            color: '#151515',
            marginBottom: '2px',
          }}>
            {template.name}
          </div>
          {/* Description */}
          {template.description && (
            <div style={{ 
              fontSize: '12px', 
              color: '#6a6e73',
              lineHeight: '1.4',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}>
              {template.description}
            </div>
          )}
        </div>
      </div>
    );
  };

  const filteredTemplates = filterTemplates(flowTemplates);

  return (
    <div
      style={{
        width: '300px',
        background: '#f5f5f5',
        borderRight: '1px solid #d2d2d2',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Tabs Header */}
      <div style={{ borderBottom: '1px solid #d2d2d2' }}>
        <Tabs
          activeKey={activeTab}
          onSelect={handleTabChange}
          aria-label="Node sidebar tabs"
          isFilled
        >
          <Tab eventKey={0} title={<TabTitleText>Node Library</TabTitleText>} data-tour="node-library-tab" />
          <Tab eventKey={1} title={<TabTitleText>Flow Templates</TabTitleText>} data-tour="flow-templates-tab" />
        </Tabs>
      </div>

      {/* Search */}
      <div style={{ padding: '12px', borderBottom: '1px solid #d2d2d2' }}>
        <SearchInput
          placeholder={activeTab === 0 ? "Search nodes..." : "Search templates..."}
          value={searchValue}
          onChange={(event, value) => setSearchValue(value)}
          onClear={() => setSearchValue('')}
          aria-label="Search"
        />
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
        {activeTab === 0 ? (
          /* Node Library Tab */
          <>
            {Object.entries(NODE_CATEGORIES).map(([categoryId, category]) => {
              const filteredNodes = filterNodes(category.nodes);
              if (filteredNodes.length === 0) return null;

              return (
                <div key={categoryId} style={{ marginBottom: '20px' }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px',
                    marginBottom: '10px',
                    paddingBottom: '6px',
                    borderBottom: '1px solid #d2d2d2',
                  }}>
                    <strong style={{ fontSize: '13px', color: '#151515' }}>
                      {category.label}
                    </strong>
                    <Badge isRead>{filteredNodes.length}</Badge>
                  </div>
                  <div>
                    {filteredNodes.map(renderNodeItem)}
                  </div>
                </div>
              );
            })}

            {searchValue && Object.values(NODE_CATEGORIES).every(cat => 
              filterNodes(cat.nodes).length === 0
            ) && (
              <div style={{ textAlign: 'center', padding: '24px', color: '#6a6e73' }}>
                <InfoCircleIcon style={{ marginBottom: '8px', fontSize: '24px' }} />
                <p>No nodes match "{searchValue}"</p>
              </div>
            )}
          </>
        ) : (
          /* Flow Templates Tab */
          <>
            {loadingTemplates ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Spinner size="lg" />
                <p style={{ marginTop: '12px', color: '#6a6e73' }}>Loading templates...</p>
              </div>
            ) : templatesError ? (
              <div style={{ textAlign: 'center', padding: '24px', color: '#c9190b' }}>
                <InfoCircleIcon style={{ marginBottom: '8px', fontSize: '24px' }} />
                <p>Error loading templates</p>
                <Button variant="link" onClick={fetchFlowTemplates}>Retry</Button>
              </div>
            ) : filteredTemplates.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px', color: '#6a6e73' }}>
                <InfoCircleIcon style={{ marginBottom: '8px', fontSize: '24px' }} />
                <p>{searchValue ? `No templates match "${searchValue}"` : 'No templates available'}</p>
              </div>
            ) : (
              <>
                {/* Group templates by category - matching Node Library style */}
                {(() => {
                  // Group filtered templates by category
                  const grouped = {};
                  filteredTemplates.forEach(template => {
                    const category = categorizeFlow(template.name, template.tags || []);
                    if (!grouped[category]) {
                      grouped[category] = [];
                    }
                    grouped[category].push(template);
                  });
                  
                  // Sort categories by order
                  const sortedCategories = Object.keys(grouped).sort((a, b) => {
                    const orderA = FLOW_CATEGORIES[a]?.order || 99;
                    const orderB = FLOW_CATEGORIES[b]?.order || 99;
                    return orderA - orderB;
                  });
                  
                  let globalTemplateIndex = 0;
                  return sortedCategories.map(categoryId => {
                    const category = FLOW_CATEGORIES[categoryId] || { name: categoryId, icon: CatalogIcon, color: '#6a6e73' };
                    const categoryTemplates = grouped[categoryId] || [];
                    const CategoryIcon = category.icon;
                    
                    return (
                      <div key={categoryId} style={{ marginBottom: '20px' }}>
                        {/* Category Header - matching Node Library style */}
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '8px',
                          marginBottom: '10px',
                          paddingBottom: '6px',
                          borderBottom: '1px solid #d2d2d2',
                        }}>
                          <strong style={{ fontSize: '13px', color: '#151515' }}>
                            {category.name}
                          </strong>
                          <Badge isRead>{categoryTemplates.length}</Badge>
                        </div>
                        
                        {/* Category Templates */}
                        <div>
                          {categoryTemplates.map(template => {
                            const isFirst = globalTemplateIndex === 0;
                            globalTemplateIndex++;
                            return renderTemplateItem(template, category.color, CategoryIcon, isFirst);
                          })}
                        </div>
                      </div>
                    );
                  });
                })()}
              </>
            )}
          </>
        )}
      </div>

      {/* Help Footer */}
      <div style={{ 
        padding: '12px 16px', 
        borderTop: '1px solid #d2d2d2',
        background: '#fff',
        fontSize: '12px',
        color: '#6a6e73',
      }}>
        {activeTab === 0 ? (
          <>
            <strong>Tip:</strong> Drag nodes onto the canvas. Press Delete to remove selected node.
          </>
        ) : (
          <>
            <strong>Tip:</strong> Load a template to start with a pre-built flow structure.
          </>
        )}
      </div>
    </div>
  );
};

export default NodeSidebar;
