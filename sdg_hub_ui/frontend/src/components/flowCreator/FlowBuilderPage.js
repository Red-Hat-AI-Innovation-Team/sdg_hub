import React, { useState, useEffect } from 'react';
import {
  PageSection,
  Title,
  Button,
  Card,
  CardTitle,
  CardBody,
  Grid,
  GridItem,
  List,
  ListItem,
  Badge,
  Toolbar,
  ToolbarContent,
  ToolbarItem,
  Alert,
  AlertVariant,
  Modal,
  ModalVariant,
  DragDrop,
  Droppable,
  Draggable,
  EmptyState,
  EmptyStateIcon,
  EmptyStateHeader,
  EmptyStateBody,
} from '@patternfly/react-core';
import {
  PlusCircleIcon,
  TrashIcon,
  EditIcon,
  GripVerticalIcon,
  ArrowLeftIcon,
  CubesIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from '@patternfly/react-icons';
import BundlesCard from './BundlesCard';
import BlockLibrary from './BlockLibrary';
import BlockConfigModal from './BlockConfigModal';
import MetadataFormModal from './MetadataFormModal';

/**
 * Flow Builder Page
 * 
 * Main interface for building custom flows with:
 * - Block list (left side) - shows current blocks in flow
 * - Block library (right side) - bundles, configured blocks, custom blocks
 */
const FlowBuilderPage = ({ initialFlow, onBack, onSave, onDraftChange, triggerSave, autoSaveOnNext }) => {
  const [blocks, setBlocks] = useState(initialFlow?.blocks || []);
  const [selectedBlockIndex, setSelectedBlockIndex] = useState(null);
  const [showBlockConfig, setShowBlockConfig] = useState(false);
  const [showMetadataForm, setShowMetadataForm] = useState(false);
  const [blockToAdd, setBlockToAdd] = useState(null);
  const [flowMetadata, setFlowMetadata] = useState(initialFlow?.metadata || {});
  const [tempFlowName, setTempFlowName] = useState(null); // Track temp flow for prompts
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // Track if changes need saving
  const [lastSavedBlocks, setLastSavedBlocks] = useState(null); // Track last saved state
  
  // Determine if we're editing an existing flow (has originalFlowName or isEditing flag)
  const existingFlowName = initialFlow?.originalFlowName || 
    (initialFlow?.isEditing ? initialFlow?.metadata?.name : null) ||
    (initialFlow?.isCloning ? null : null); // For cloning, we create new prompts
  
  // Check if we're in edit mode (editing existing custom flow)
  const isEditMode = initialFlow?.isEditing || initialFlow?.originalFlowName;

  /**
   * Generate unique ID for blocks
   */
  const generateBlockId = () => {
    return `block_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  /**
   * Auto-save draft whenever blocks or metadata change
   */
  useEffect(() => {
    if (onDraftChange && (blocks.length > 0 || flowMetadata.name)) {
      onDraftChange({
        blocks,
        metadata: flowMetadata,
        tempFlowName
      });
    }
  }, [blocks, flowMetadata, tempFlowName]);

  /**
   * Update tempFlowName when initialFlow changes (restore from draft)
   */
  useEffect(() => {
    if (initialFlow?.tempFlowName) {
      setTempFlowName(initialFlow.tempFlowName);
    }
  }, [initialFlow?.tempFlowName]);

  /**
   * Update blocks when initialFlow changes (e.g., when cloning)
   */
  useEffect(() => {
    if (initialFlow?.blocks) {
      // Ensure all blocks have unique IDs
      const blocksWithIds = initialFlow.blocks.map(block => ({
        ...block,
        _id: block._id || generateBlockId()
      }));
      setBlocks(blocksWithIds);
      // Store initial blocks as last saved state for comparison
      setLastSavedBlocks(JSON.stringify(blocksWithIds));
    }
    if (initialFlow?.metadata) {
      setFlowMetadata(initialFlow.metadata);
    }
  }, [initialFlow]);

  /**
   * Track unsaved changes by comparing current blocks to last saved state
   */
  useEffect(() => {
    if (lastSavedBlocks !== null && blocks.length > 0) {
      const currentBlocksStr = JSON.stringify(blocks);
      const hasChanges = currentBlocksStr !== lastSavedBlocks;
      setHasUnsavedChanges(hasChanges);
    } else if (blocks.length > 0 && !isEditMode) {
      // New flow with blocks - always has unsaved changes
      setHasUnsavedChanges(true);
    }
  }, [blocks, lastSavedBlocks, isEditMode]);

  /**
   * Handle adding a block from the library
   */
  const handleAddBlock = (blockTemplate) => {
    setBlockToAdd(blockTemplate);
    setShowBlockConfig(true);
  };

  /**
   * Handle block configuration submit
   */
  const handleBlockConfigSubmit = (configuredBlock) => {
    
    if (selectedBlockIndex !== null) {
      // Editing existing block - preserve its ID and merge with existing properties
      setBlocks(prev => prev.map((b, i) => {
        if (i === selectedBlockIndex) {
          // Preserve _id and merge the updated config
          return {
            ...configuredBlock,
            _id: b._id, // Preserve the block's unique ID
          };
        }
        return b;
      }));
    } else {
      // Adding new block(s)
      if (Array.isArray(configuredBlock)) {
        // Bundle expanded to multiple blocks - add unique IDs
        const blocksWithIds = configuredBlock.map(block => ({
          ...block,
          _id: generateBlockId()
        }));
        setBlocks(prev => {
          const newBlocks = [...prev, ...blocksWithIds];
          return newBlocks;
        });
      } else {
        // Single block - add unique ID
        const blockWithId = {
          ...configuredBlock,
          _id: generateBlockId()
        };
        setBlocks(prev => [...prev, blockWithId]);
      }
    }
    setShowBlockConfig(false);
    setBlockToAdd(null);
    setSelectedBlockIndex(null);
  };

  /**
   * Handle editing an existing block
   */
  const handleEditBlock = (index) => {
    setSelectedBlockIndex(index);
    setBlockToAdd(blocks[index]);
    setShowBlockConfig(true);
  };

  /**
   * Handle deleting a block
   */
  const handleDeleteBlock = (index) => {
    setBlocks(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Handle reordering blocks via drag and drop
   */
  const handleDrop = (source, dest) => {
    if (!dest || source.index === dest.index) return;
    
    const newBlocks = [...blocks];
    const [removed] = newBlocks.splice(source.index, 1);
    newBlocks.splice(dest.index, 0, removed);
    setBlocks(newBlocks);
  };

  /**
   * Move block up
   */
  const moveBlockUp = (index) => {
    if (index === 0) return;
    const newBlocks = [...blocks];
    [newBlocks[index - 1], newBlocks[index]] = [newBlocks[index], newBlocks[index - 1]];
    setBlocks(newBlocks);
  };

  /**
   * Move block down
   */
  const moveBlockDown = (index) => {
    if (index === blocks.length - 1) return;
    const newBlocks = [...blocks];
    [newBlocks[index], newBlocks[index + 1]] = [newBlocks[index + 1], newBlocks[index]];
    setBlocks(newBlocks);
  };

  /**
   * Handle save button click
   */
  const handleSaveClick = () => {
    if (blocks.length === 0) {
      alert('Please add at least one block to your flow before saving.');
      return;
    }
    setShowMetadataForm(true);
  };

  /**
   * Expose save function to parent via ref
   */
  useEffect(() => {
    if (triggerSave) {
      triggerSave.current = handleSaveClick;
    }
  }, [triggerSave]);

  /**
   * Notify parent when save is needed on next navigation
   */
  useEffect(() => {
    if (autoSaveOnNext && blocks.length > 0) {
      // Tell parent that this flow needs to be saved before proceeding
      autoSaveOnNext({
        needsSave: true,
        blockCount: blocks.length,
        openMetadataModal: handleSaveClick,
        hasUnsavedChanges: hasUnsavedChanges,
        isEditMode: isEditMode,
        flowName: flowMetadata?.name,
        triggerQuickSave: handleQuickSave
      });
    } else if (autoSaveOnNext) {
      // No blocks or already saved
      autoSaveOnNext({ needsSave: false });
    }
  }, [autoSaveOnNext, blocks.length, hasUnsavedChanges, isEditMode, flowMetadata?.name]);

  /**
   * Handle metadata submission and final save
   */
  const handleMetadataSubmit = async (metadata) => {
    setFlowMetadata(metadata);
    setShowMetadataForm(false);
    
    // Call parent save function with complete flow
    const completeFlow = {
      metadata,
      blocks,
      tempFlowName: tempFlowName // Pass temp flow name for prompt file copying
    };
    
    if (onSave) {
      await onSave(completeFlow);
      // Update last saved state after successful save
      setLastSavedBlocks(JSON.stringify(blocks));
      setHasUnsavedChanges(false);
    }
  };

  /**
   * Handle quick save for edit mode (re-save with existing metadata)
   */
  const handleQuickSave = async () => {
    if (!flowMetadata.name) {
      // No metadata yet, need to show the form
      setShowMetadataForm(true);
      return;
    }
    
    // Call parent save function with existing metadata
    const completeFlow = {
      metadata: flowMetadata,
      blocks,
      tempFlowName: tempFlowName
    };
    
    if (onSave) {
      await onSave(completeFlow);
      // Update last saved state after successful save
      setLastSavedBlocks(JSON.stringify(blocks));
      setHasUnsavedChanges(false);
    }
  };

  return (
    <PageSection>
      {/* Header with Save Button */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '1.5rem'
      }}>
        <div>
          <Button variant="link" icon={<ArrowLeftIcon />} onClick={onBack}>
            Back to Template Selection
          </Button>
          <Title headingLevel="h1" size="2xl" style={{ marginTop: '0.5rem' }}>
            Flow Builder
            {isEditMode && flowMetadata?.name && (
              <span style={{ fontSize: '1rem', fontWeight: 'normal', color: '#6a6e73', marginLeft: '12px' }}>
                - Editing: {flowMetadata.name}
              </span>
            )}
          </Title>
          <p style={{ color: '#6a6e73', marginTop: '0.5rem' }}>
            Build your flow by adding blocks and bundles from the library
          </p>
        </div>
      </div>

      {/* Main Builder Interface */}
      <Grid hasGutter>
        {/* Left Side - Current Blocks */}
        <GridItem span={8}>
          <Card isFullHeight>
            <CardTitle>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title headingLevel="h2" size="xl">
                  Flow Blocks ({blocks.length})
                </Title>
                {blocks.length > 0 && (
                  <Badge isRead>{blocks.length} blocks</Badge>
                )}
              </div>
            </CardTitle>
            <CardBody>
              {blocks.length === 0 ? (
                <EmptyState>
                  <EmptyStateHeader 
                    titleText="No blocks yet" 
                    icon={<EmptyStateIcon icon={CubesIcon} />}
                    headingLevel="h3"
                  />
                  <EmptyStateBody>
                    Add blocks from the library on the right to start building your flow.
                    You can use pre-configured bundles or add individual blocks.
                  </EmptyStateBody>
                </EmptyState>
              ) : (
                <DragDrop onDrop={handleDrop}>
                  <Droppable zone="blocks-zone" hasNoWrapper>
                    <List isPlain isBordered>
                      {blocks.map((block, index) => (
                        <Draggable key={block._id || index} zone="blocks-zone" hasNoWrapper>
                          <ListItem>
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              padding: '0.75rem',
                              background: '#f5f5f5',
                              borderRadius: '4px',
                              cursor: 'default'
                            }}>
                              <div 
                                style={{ 
                                  color: '#6a6e73', 
                                  cursor: 'grab',
                                  padding: '0.25rem',
                                  display: 'flex',
                                  alignItems: 'center'
                                }}
                                title="Drag to reorder"
                              >
                                <GripVerticalIcon />
                              </div>
                              <Badge isRead>{index + 1}</Badge>
                              <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: 'bold' }}>
                                  {block.block_config?.block_name || block.block_name || `Block ${index + 1}`}
                                </div>
                                <div style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
                                  {block.block_type}
                                </div>
                              </div>
                              <div style={{ display: 'flex', gap: '0.25rem' }}>
                                <Button 
                                  variant="plain" 
                                  icon={<ArrowUpIcon />}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    moveBlockUp(index);
                                  }}
                                  aria-label="Move block up"
                                  isDisabled={index === 0}
                                  size="sm"
                                />
                                <Button 
                                  variant="plain" 
                                  icon={<ArrowDownIcon />}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    moveBlockDown(index);
                                  }}
                                  aria-label="Move block down"
                                  isDisabled={index === blocks.length - 1}
                                  size="sm"
                                />
                              </div>
                              <Button 
                                variant="plain" 
                                icon={<EditIcon />}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleEditBlock(index);
                                }}
                                aria-label="Edit block"
                                size="sm"
                              />
                              <Button 
                                variant="plain" 
                                icon={<TrashIcon />}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteBlock(index);
                                }}
                                aria-label="Delete block"
                                isDanger
                                size="sm"
                              />
                            </div>
                          </ListItem>
                        </Draggable>
                      ))}
                    </List>
                  </Droppable>
                </DragDrop>
              )}
            </CardBody>
          </Card>
        </GridItem>

        {/* Right Side - Bundles and Block Library */}
        <GridItem span={4}>
          <Grid hasGutter>
            {/* Bundles Card */}
            <GridItem span={12}>
              <BundlesCard onAddBundle={handleAddBlock} />
            </GridItem>
            
            {/* Block Library Card */}
            <GridItem span={12}>
              <BlockLibrary onAddBlock={handleAddBlock} />
            </GridItem>
          </Grid>
        </GridItem>
      </Grid>

      {/* Block Configuration Modal */}
      {showBlockConfig && blockToAdd && (
        <BlockConfigModal
          block={blockToAdd}
          isEdit={selectedBlockIndex !== null}
          onSubmit={handleBlockConfigSubmit}
          onClose={() => {
            setShowBlockConfig(false);
            setBlockToAdd(null);
            setSelectedBlockIndex(null);
          }}
          onTempFlowCreated={(tempName) => {
            setTempFlowName(tempName);
          }}
          existingFlowName={existingFlowName || flowMetadata?.name || tempFlowName}
        />
      )}

      {/* Metadata Form Modal */}
      {showMetadataForm && (
        <MetadataFormModal
          initialMetadata={flowMetadata}
          onSubmit={handleMetadataSubmit}
          onClose={() => setShowMetadataForm(false)}
        />
      )}
    </PageSection>
  );
};

export default FlowBuilderPage;

