import React, { useState, useEffect } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Accordion,
  AccordionItem,
  AccordionContent,
  AccordionToggle,
  List,
  ListItem,
  Button,
  Badge,
  SearchInput,
  Spinner,
  Alert,
  AlertVariant,
} from '@patternfly/react-core';
import { PlusCircleIcon } from '@patternfly/react-icons';
import { blockAPI, API_BASE_URL } from '../../services/api';

/**
 * Block Library Component
 * 
 * Shows available blocks organized into:
 * 1. Bundles - Pre-configured block sequences
 * 2. Configured Blocks - Common block configurations
 * 3. Custom Blocks - All available block types from SDG Hub
 */
const BlockLibrary = ({ onAddBlock }) => {
  const [expandedSections, setExpandedSections] = useState(['bundles']);
  const [searchValue, setSearchValue] = useState('');
  const [discoveredBlocks, setDiscoveredBlocks] = useState([]);
  const [configuredBlocks, setConfiguredBlocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Load available blocks and templates from backend on mount
   */
  useEffect(() => {
    loadBlocks();
    loadConfiguredBlocks();
  }, []);

  const loadBlocks = async () => {
    try {
      setLoading(true);
      const response = await blockAPI.listBlocks();
      setDiscoveredBlocks(response.blocks || []);
    } catch (err) {
      setError('Failed to load blocks from SDG Hub: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadConfiguredBlocks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/blocks/templates`);
      const data = await response.json();
      setConfiguredBlocks(data.templates || []);
    } catch (err) {
      console.warn('Failed to load configured blocks:', err.message);
    }
  };

  /**
   * Toggle section expansion
   */
  const toggleSection = (section) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  /**
   * Individual Block Types - Hardcoded with nice defaults
   */
  const AVAILABLE_BLOCKS = [
    // LLM Blocks
    { 
      id: 'LLMChatBlock', 
      name: 'LLM Chat', 
      category: 'LLM',
      description: 'Generate text using LLM',
      icon: '🤖',
      defaultConfig: {
        block_type: 'LLMChatBlock',
        block_config: {
          block_name: 'llm_chat',
          input_cols: 'prompt',
          output_cols: 'response',
          max_tokens: 2048,
          temperature: 0.7,
          async_mode: true,
        }
      }
    },
    { 
      id: 'PromptBuilderBlock', 
      name: 'Prompt Builder', 
      category: 'LLM',
      description: 'Build prompts from template',
      icon: '📋',
      defaultConfig: {
        block_type: 'PromptBuilderBlock',
        block_config: {
          block_name: 'prompt_builder',
          input_cols: ['text'],
          output_cols: 'prompt',
          prompt_config_path: 'custom_prompt.yaml',
        }
      }
    },
    { 
      id: 'LLMParserBlock', 
      name: 'LLM Parser', 
      category: 'LLM',
      description: 'Parse LLM responses',
      icon: '🔍',
      defaultConfig: {
        block_type: 'LLMParserBlock',
        block_config: {
          block_name: 'llm_parser',
          input_cols: 'raw_response',
          extract_content: true,
        }
      }
    },
    { 
      id: 'TextParserBlock', 
      name: 'Text Parser', 
      category: 'Transform',
      description: 'Extract text with tags',
      icon: '✂️',
      defaultConfig: {
        block_type: 'TextParserBlock',
        block_config: {
          block_name: 'text_parser',
          input_cols: 'content',
          output_cols: 'extracted',
          start_tags: [''],
          end_tags: [''],
        }
      }
    },
    // Transform Blocks
    { 
      id: 'DuplicateColumnsBlock', 
      name: 'Duplicate Columns', 
      category: 'Transform',
      description: 'Duplicate dataset columns',
      icon: '📑',
      defaultConfig: {
        block_type: 'DuplicateColumnsBlock',
        block_config: {
          block_name: 'duplicate_columns',
          input_cols: { column: 'column_copy' },
        }
      }
    },
    { 
      id: 'RenameColumnsBlock', 
      name: 'Rename Columns', 
      category: 'Transform',
      description: 'Rename dataset columns',
      icon: '✏️',
      defaultConfig: {
        block_type: 'RenameColumnsBlock',
        block_config: {
          block_name: 'rename_columns',
          input_cols: { old_name: 'new_name' },
        }
      }
    },
    { 
      id: 'MeltColumnsBlock', 
      name: 'Melt Columns', 
      category: 'Transform',
      description: 'Unpivot columns into rows',
      icon: '🔄',
      defaultConfig: {
        block_type: 'MeltColumnsBlock',
        block_config: {
          block_name: 'melt_columns',
          input_cols: ['col1', 'col2'],
          output_cols: ['value', 'type'],
        }
      }
    },
    { 
      id: 'JSONStructureBlock', 
      name: 'JSON Structure', 
      category: 'Transform',
      description: 'Create structured JSON output',
      icon: '{ }',
      defaultConfig: {
        block_type: 'JSONStructureBlock',
        block_config: {
          block_name: 'json_structure',
          input_cols: ['field1', 'field2'],
          output_cols: ['structured_output'],
          ensure_json_serializable: true,
        }
      }
    },
    // Filter Blocks
    { 
      id: 'ColumnValueFilterBlock', 
      name: 'Column Value Filter', 
      category: 'Filter',
      description: 'Filter rows by column value',
      icon: '🔽',
      defaultConfig: {
        block_type: 'ColumnValueFilterBlock',
        block_config: {
          block_name: 'filter',
          input_cols: ['column'],
          filter_value: 'value',
          operation: 'eq',
        }
      }
    },
  ];

  /**
   * Convert discovered blocks to display format
   */
  const convertDiscoveredBlocksToDisplay = () => {
    return discoveredBlocks.map(blockName => {
      // Try to find in our predefined list first
      const predefined = AVAILABLE_BLOCKS.find(b => b.id === blockName);
      if (predefined) return predefined;
      
      // Otherwise create a basic entry
      return {
        id: blockName,
        name: blockName.replace('Block', ''),
        category: blockName.includes('LLM') ? 'LLM' : blockName.includes('Filter') ? 'Filter' : 'Transform',
        description: `${blockName} from SDG Hub`,
        icon: '🔧',
        defaultConfig: {
          block_type: blockName,
          block_config: {
            block_name: blockName.toLowerCase(),
            input_cols: [],
            output_cols: [],
          }
        }
      };
    });
  };

  /**
   * Combine hardcoded and discovered blocks
   */
  const allBlocks = [
    ...AVAILABLE_BLOCKS,
    ...convertDiscoveredBlocksToDisplay().filter(db => 
      !AVAILABLE_BLOCKS.find(ab => ab.id === db.id)
    )
  ];

  /**
   * Filter blocks by search
   */
  const filteredBlocks = allBlocks.filter(block =>
    block.name.toLowerCase().includes(searchValue.toLowerCase()) ||
    block.description.toLowerCase().includes(searchValue.toLowerCase()) ||
    block.id.toLowerCase().includes(searchValue.toLowerCase())
  );

  /**
   * Group blocks by category
   */
  const blocksByCategory = filteredBlocks.reduce((acc, block) => {
    if (!acc[block.category]) {
      acc[block.category] = [];
    }
    acc[block.category].push(block);
    return acc;
  }, {});

  if (loading) {
    return (
      <Card isFullHeight>
        <CardTitle>
          <Title headingLevel="h2" size="xl">
            Block Library
          </Title>
        </CardTitle>
        <CardBody style={{ textAlign: 'center', padding: '2rem' }}>
          <Spinner size="xl" />
          <p style={{ marginTop: '1rem', color: '#6a6e73' }}>Loading blocks from SDG Hub...</p>
        </CardBody>
      </Card>
    );
  }

  if (error) {
    return (
      <Card isFullHeight>
        <CardTitle>
          <Title headingLevel="h2" size="xl">
            Block Library
          </Title>
        </CardTitle>
        <CardBody>
          <Alert variant={AlertVariant.danger} isInline title="Failed to load blocks">
            {error}
          </Alert>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card isFullHeight>
      <CardTitle>
        <Title headingLevel="h2" size="xl">
          Block Library
        </Title>
        <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginTop: '0.25rem' }}>
          {discoveredBlocks.length} blocks from SDG Hub
        </div>
      </CardTitle>
      <CardBody style={{ padding: '1rem' }}>
        {/* Search */}
        <SearchInput
          placeholder="Search blocks..."
          value={searchValue}
          onChange={(event, value) => setSearchValue(value)}
          onClear={() => setSearchValue('')}
          style={{ marginBottom: '1rem' }}
        />

        <Accordion asDefinitionList>
          {/* Configured Blocks Section */}
          <AccordionItem>
            <AccordionToggle
              onClick={() => toggleSection('configured')}
              isExpanded={expandedSections.includes('configured')}
              id="configured-toggle"
            >
              <strong>🎨 Configured Blocks</strong>
              <Badge isRead style={{ marginLeft: '0.5rem' }}>{configuredBlocks.length}</Badge>
            </AccordionToggle>
            <AccordionContent
              id="configured-content"
              isHidden={!expandedSections.includes('configured')}
            >
              <List isPlain style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {configuredBlocks
                  .filter(template => 
                    !searchValue || 
                    template.name.toLowerCase().includes(searchValue.toLowerCase()) ||
                    template.type.toLowerCase().includes(searchValue.toLowerCase())
                  )
                  .map((template, index) => (
                  <ListItem key={index}>
                    <div style={{
                      padding: '0.5rem',
                      background: '#f5f5f5',
                      borderRadius: '4px',
                      marginBottom: '0.5rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>
                          {template.name}
                        </div>
                        <div style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                          {template.type}
                        </div>
                      </div>
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={<PlusCircleIcon />}
                        onClick={() => onAddBlock({
                          ...template,
                          isBundle: false,
                          defaultConfig: {
                            block_type: template.type,
                            block_config: template.config
                          }
                        })}
                      >
                        Add
                      </Button>
                    </div>
                  </ListItem>
                ))}
              </List>
            </AccordionContent>
          </AccordionItem>

          {/* Individual Blocks by Category */}
          {Object.keys(blocksByCategory).map(category => (
            <AccordionItem key={category}>
              <AccordionToggle
                onClick={() => toggleSection(category)}
                isExpanded={expandedSections.includes(category)}
                id={`${category}-toggle`}
              >
                <strong>{category} Blocks</strong>
                <Badge isRead style={{ marginLeft: '0.5rem' }}>{blocksByCategory[category].length}</Badge>
              </AccordionToggle>
              <AccordionContent
                id={`${category}-content`}
                isHidden={!expandedSections.includes(category)}
              >
                <List isPlain>
                  {blocksByCategory[category].map(block => (
                    <ListItem key={block.id}>
                      <div style={{
                        padding: '0.5rem',
                        background: '#f5f5f5',
                        borderRadius: '4px',
                        marginBottom: '0.5rem',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>
                            {block.icon} {block.name}
                          </div>
                          <div style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                            {block.description}
                          </div>
                        </div>
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<PlusCircleIcon />}
                          onClick={() => onAddBlock({ ...block, isBundle: false })}
                        >
                          Add
                        </Button>
                      </div>
                    </ListItem>
                  ))}
                </List>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardBody>
    </Card>
  );
};

export default BlockLibrary;

