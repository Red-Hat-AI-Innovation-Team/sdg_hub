import React, { useState, useEffect } from 'react';
import {
  PageSection,
  Title,
  Button,
} from '@patternfly/react-core';
import {
  ArrowLeftIcon,
} from '@patternfly/react-icons';
import MetadataFormModal from './MetadataFormModal';
import VisualFlowEditor from './VisualFlowEditor';

/**
 * Flow Builder Page
 * 
 * Main interface for building custom flows with:
 * - Block list (left side) - shows current blocks in flow
 * - Bundles panel (right side) - pre-configured block bundles
 */
const FlowBuilderPage = ({ initialFlow, onBack, onSave, onDraftChange, triggerSave, autoSaveOnNext, isBlankStart = false }) => {
  const [blocks, setBlocks] = useState(initialFlow?.blocks || []);
  const [showMetadataForm, setShowMetadataForm] = useState(false);
  const [flowMetadata, setFlowMetadata] = useState(initialFlow?.metadata || {});
  const [tempFlowName, setTempFlowName] = useState(null); // Track temp flow for prompts
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // Track if changes need saving
  const [lastSavedBlocks, setLastSavedBlocks] = useState(null); // Track last saved state
  const [visualNodes, setVisualNodes] = useState(initialFlow?.visualNodes || []);
  const [visualEdges, setVisualEdges] = useState(initialFlow?.visualEdges || []);
  const [sourceFlowName, setSourceFlowName] = useState(initialFlow?.sourceFlowName || null); // Track source for prompt copying
  
  // Sync visual state when initialFlow changes (e.g., when loading for editing)
  // This is needed because useState only uses initial value on first mount
  const initialFlowIdRef = React.useRef(null);
  const lastVisualNodesCountRef = React.useRef(0);
  const isLoadingRef = React.useRef(false);
  
  React.useEffect(() => {
    const flowId = initialFlow?.metadata?.id || initialFlow?.metadata?.name || initialFlow?.originalFlowName;
    const currentVisualNodesCount = initialFlow?.visualNodes?.length || 0;
    
    // Skip if no flow
    if (!initialFlow || !flowId) {
      return;
    }
    
    // Check if this is a new flow OR if visual nodes data has been loaded (count changed from 0 to > 0)
    const isNewFlow = initialFlowIdRef.current !== flowId;
    const visualNodesJustLoaded = currentVisualNodesCount > 0 && lastVisualNodesCountRef.current === 0;
    
    console.log('FlowBuilderPage: Processing initialFlow:', flowId, 'visualNodes:', currentVisualNodesCount, 'isNewFlow:', isNewFlow, 'visualNodesJustLoaded:', visualNodesJustLoaded);
    
    // AbortController to cancel stale fetch if flow changes mid-request
    let abortController;
    
    // If we have visual nodes, use them directly
    // Update if: new flow OR visual nodes were just loaded (API completed)
    if (initialFlow.visualNodes && initialFlow.visualNodes.length > 0 && (isNewFlow || visualNodesJustLoaded)) {
      setVisualNodes(initialFlow.visualNodes);
      setVisualEdges(initialFlow.visualEdges || []);
      lastVisualNodesCountRef.current = currentVisualNodesCount;
      initialFlowIdRef.current = flowId;
      console.log('FlowBuilderPage: Set', initialFlow.visualNodes.length, 'nodes from initialFlow');
    } 
    // If we have blocks but no visual nodes, try to load from backend
    else if (initialFlow.blocks?.length > 0 && currentVisualNodesCount === 0 && !isLoadingRef.current && isNewFlow) {
      const flowName = initialFlow.metadata?.name || initialFlow.originalFlowName;
      if (flowName) {
        console.log('FlowBuilderPage: Fetching visual nodes from backend for:', flowName);
        isLoadingRef.current = true;
        abortController = new AbortController();
        
        fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/flows/${encodeURIComponent(flowName)}/yaml`, {
          signal: abortController.signal,
        })
          .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
          })
          .then(flowYamlData => {
            console.log('FlowBuilderPage: Backend returned visualNodes:', flowYamlData.visualNodes?.length);
            if (flowYamlData.visualNodes && flowYamlData.visualNodes.length > 0) {
              setVisualNodes(flowYamlData.visualNodes);
              setVisualEdges(flowYamlData.visualEdges || []);
              lastVisualNodesCountRef.current = flowYamlData.visualNodes.length;
            }
            initialFlowIdRef.current = flowId;
            isLoadingRef.current = false;
          })
          .catch(error => {
            if (error.name === 'AbortError') return; // Ignore aborted requests
            console.error('FlowBuilderPage: Failed to load from backend:', error);
            initialFlowIdRef.current = flowId;
            isLoadingRef.current = false;
          });
      }
    }
    
    // Always sync blocks and metadata for new flows
    if (isNewFlow) {
      if (initialFlow.blocks) {
        setBlocks(initialFlow.blocks);
      }
      if (initialFlow.metadata) {
        // Extract required_columns from dataset_requirements if not already at top level
        const requiredColumns = initialFlow.metadata.required_columns 
          || initialFlow.metadata.dataset_requirements?.required_columns 
          || [];
        setFlowMetadata({
          ...initialFlow.metadata,
          required_columns: requiredColumns,
        });
      }
      initialFlowIdRef.current = flowId;
    }
    if (initialFlow.sourceFlowName) {
      setSourceFlowName(initialFlow.sourceFlowName);
    }
    
    // Cleanup: abort in-flight fetch if flow changes or component unmounts
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, [initialFlow]);
  
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
      // Extract required_columns from dataset_requirements if not already at top level
      const requiredColumns = initialFlow.metadata.required_columns 
        || initialFlow.metadata.dataset_requirements?.required_columns 
        || [];
      setFlowMetadata({
        ...initialFlow.metadata,
        required_columns: requiredColumns,
      });
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
    
    console.log('FlowBuilderPage handleMetadataSubmit: visualNodes state has', visualNodes?.length, 'nodes, visualEdges has', visualEdges?.length, 'edges');
    
    // Call parent save function with complete flow
    const completeFlow = {
      metadata,
      blocks,
      visualNodes: visualNodes, // Include visual nodes for editor state
      visualEdges: visualEdges, // Include visual edges for editor state
      tempFlowName: tempFlowName, // Pass temp flow name for prompt file copying
      sourceFlowName: sourceFlowName, // Pass source flow name for template prompt copying
    };
    
    console.log('FlowBuilderPage handleMetadataSubmit: completeFlow.visualNodes:', completeFlow.visualNodes?.length);
    
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
      visualNodes: visualNodes, // Include visual nodes for editor state
      visualEdges: visualEdges, // Include visual edges for editor state
      tempFlowName: tempFlowName,
      sourceFlowName: sourceFlowName, // Pass source flow name for template prompt copying
    };
    
    if (onSave) {
      await onSave(completeFlow);
      // Update last saved state after successful save
      setLastSavedBlocks(JSON.stringify(blocks));
      setHasUnsavedChanges(false);
    }
  };

  /**
   * Handle visual editor save
   */
  const handleVisualEditorSave = async (completeFlow) => {
    // Extract blocks from visual flow
    if (completeFlow.blocks) {
      setBlocks(completeFlow.blocks);
    }
    if (completeFlow.visualNodes) {
      setVisualNodes(completeFlow.visualNodes);
    }
    if (completeFlow.visualEdges) {
      setVisualEdges(completeFlow.visualEdges);
    }
    if (completeFlow.metadata) {
      setFlowMetadata(completeFlow.metadata);
    }
    // Track source flow for prompt copying
    if (completeFlow.sourceFlowName) {
      setSourceFlowName(completeFlow.sourceFlowName);
    }

    // Show metadata form for final save
    setShowMetadataForm(true);
  };

  /**
   * Handle visual editor draft change
   */
  const handleVisualDraftChange = (draft) => {
    // If the editor was cleared, reset local state completely
    if (draft.cleared) {
      setVisualNodes([]);
      setVisualEdges([]);
      setBlocks([]);
      setFlowMetadata({});
      if (onDraftChange) {
        onDraftChange({ visualNodes: [], visualEdges: [], blocks: [], metadata: {}, cleared: true });
      }
      return;
    }
    if (draft.visualNodes) {
      setVisualNodes(draft.visualNodes);
    }
    if (draft.visualEdges) {
      setVisualEdges(draft.visualEdges);
    }
    if (onDraftChange) {
      onDraftChange({
        ...draft,
        blocks,
        metadata: flowMetadata,
        tempFlowName,
      });
    }
  };

  // Calculate height: viewport height minus page header (~140px) and wizard footer (~80px)
  const calculatedHeight = 'calc(100vh - 220px)';
  
  return (
    <PageSection padding={{ default: 'noPadding' }} style={{ height: calculatedHeight, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Compact Header */}
      <div style={{ 
        padding: '8px 16px', 
        borderBottom: '1px solid #d2d2d2',
        background: '#fff',
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        flexShrink: 0,
      }}>
        <Button variant="link" icon={<ArrowLeftIcon />} onClick={onBack} style={{ padding: 0 }}>
          Back
        </Button>
        <Title headingLevel="h1" size="lg" style={{ margin: 0 }}>
          Flow Builder
          {isEditMode && flowMetadata?.name && (
            <span style={{ fontSize: '0.875rem', fontWeight: 'normal', color: '#6a6e73', marginLeft: '12px' }}>
              - Editing: {flowMetadata.name}
            </span>
          )}
        </Title>
      </div>

      {/* Visual Flow Editor - fills remaining space */}
      <div style={{ 
        flex: 1,
        overflow: 'hidden',
        minHeight: 0, // Required for flex child to shrink properly
      }}>
        <VisualFlowEditor
          initialFlow={{
            visualNodes,
            visualEdges,
            metadata: flowMetadata,
            blocks,
            isCloning: initialFlow?.isCloning,
            sourceFlowName: sourceFlowName || initialFlow?.sourceFlowName,
          }}
          onSave={handleVisualEditorSave}
          onBack={onBack}
          onDraftChange={handleVisualDraftChange}
          existingFlowName={existingFlowName}
          isEditMode={isEditMode}
          isBlankStart={isBlankStart}
        />
      </div>

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

