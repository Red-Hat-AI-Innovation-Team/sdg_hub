import React, { useState, useEffect } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  TextInput,
  SearchInput,
  MenuToggle,
  Select,
  SelectOption,
  SelectList,
  Grid,
  GridItem,
  Button,
  Badge,
  Chip,
  ChipGroup,
  Spinner,
  EmptyState,
  EmptyStateIcon,
  EmptyStateBody,
  List,
  ListItem,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  FileUpload,
  Alert,
  AlertVariant,
  Divider,
} from '@patternfly/react-core';
import { SearchIcon, CheckCircleIcon, UploadIcon } from '@patternfly/react-icons';
import { flowAPI, configAPI } from '../../services/api';

/**
 * Flow Selection Step Component
 * 
 * Allows users to:
 * - Browse all available flows
 * - Search flows by tag
 * - View flow details
 * - Select a flow for configuration
 */
const FlowSelectionStep = ({ selectedFlow, onFlowSelect, onError, isImported }) => {
  const [flows, setFlows] = useState([]);
  const [filteredFlows, setFilteredFlows] = useState([]);
  const [selectedFlowDetails, setSelectedFlowDetails] = useState(null);
  const [searchValue, setSearchValue] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [isTagSelectOpen, setIsTagSelectOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);


  // Available tags (hardcoded for now, could be fetched from API)
  const availableTags = [
    'question-generation',
    'knowledge-extraction',
    'qa-pairs',
    'document-processing',
    'educational',
    'text-analysis',
    'sentiment-analysis',
  ];

  /**
   * Load all flows on component mount
   */
  useEffect(() => {
    loadFlows();
  }, []);

  /**
   * Filter flows when search or tags change
   */
  useEffect(() => {
    filterFlows();
  }, [flows, searchValue, selectedTags]);

  /**
   * Load flows from API
   */
  const loadFlows = async () => {
    try {
      setLoading(true);
      const flowList = await flowAPI.listFlows();
      setFlows(flowList);
      setFilteredFlows(flowList);
    } catch (error) {
      onError('Failed to load flows: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Separate flows into SDG Hub and Custom
   */
  const separateFlows = (flowList) => {
    const sdgHub = flowList.filter(flow => !flow.includes('(Custom)'));
    const custom = flowList.filter(flow => flow.includes('(Custom)'));
    return { sdgHub, custom };
  };

  /**
   * Filter flows based on search and tags
   */
  const filterFlows = () => {
    let filtered = [...flows];

    // Apply search filter
    if (searchValue) {
      filtered = filtered.filter((flow) =>
        flow.toLowerCase().includes(searchValue.toLowerCase())
      );
    }

    // Apply tag filter (if any tags selected)
    if (selectedTags.length > 0) {
      // For now, just filter by search
      // In a full implementation, we'd call the API with tag filters
    }

    setFilteredFlows(filtered);
  };

  /**
   * Handle flow selection
   */
  const handleFlowClick = async (flowName) => {
    try {
      setLoadingDetails(true);
      
      // Get detailed flow information
      const flowInfo = await flowAPI.getFlowInfo(flowName);
      setSelectedFlowDetails(flowInfo);
      
      // Select the flow in the backend
      await flowAPI.selectFlow(flowName);
      
      // Notify parent component immediately (this saves it to wizard state)
      onFlowSelect(flowInfo);
      
    } catch (error) {
      onError('Failed to load flow details: ' + error.message);
    } finally {
      setLoadingDetails(false);
    }
  };
  
  /**
   * Restore selected flow details when coming back to this step
   */
  useEffect(() => {
    if (selectedFlow && !selectedFlowDetails) {
      setSelectedFlowDetails(selectedFlow);
    }
  }, [selectedFlow, selectedFlowDetails]);

  /**
   * Handle tag selection
   */
  const handleTagSelect = (event, selection) => {
    if (selectedTags.includes(selection)) {
      setSelectedTags(selectedTags.filter((tag) => tag !== selection));
    } else {
      setSelectedTags([...selectedTags, selection]);
    }
  };

  /**
   * Clear all filters
   */
  const handleClearFilters = () => {
    setSearchValue('');
    setSelectedTags([]);
  };


  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem' }}>
        <Spinner size="xl" />
        <div style={{ marginTop: '1rem' }}>Loading available flows...</div>
      </div>
    );
  }

  return (
    <Grid hasGutter style={{ height: '100%' }}>
      {/* Left Panel - Flow List */}
      <GridItem span={6} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Available Flows
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* Search and Filter */}
            <div style={{ marginBottom: '1rem' }}>
              <SearchInput
                placeholder="Search flows..."
                value={searchValue}
                onChange={(event, value) => setSearchValue(value)}
                onClear={() => setSearchValue('')}
                style={{ marginBottom: '0.5rem' }}
              />
              
              <Select
                toggle={(toggleRef) => (
                  <MenuToggle
                    ref={toggleRef}
                    onClick={() => setIsTagSelectOpen(!isTagSelectOpen)}
                    isExpanded={isTagSelectOpen}
                  >
                    {selectedTags.length > 0 ? `${selectedTags.length} tags selected` : 'Filter by tags'}
                  </MenuToggle>
                )}
                isOpen={isTagSelectOpen}
                onOpenChange={(isOpen) => setIsTagSelectOpen(isOpen)}
                onSelect={(event, selection) => handleTagSelect(event, selection)}
                selected={selectedTags}
                style={{ marginBottom: '0.5rem' }}
              >
                <SelectList>
                  {availableTags.map((tag) => (
                    <SelectOption key={tag} value={tag}>
                      {tag}
                    </SelectOption>
                  ))}
                </SelectList>
              </Select>

              {(searchValue || selectedTags.length > 0) && (
                <Button
                  variant="link"
                  onClick={handleClearFilters}
                  style={{ padding: 0 }}
                >
                  Clear filters
                </Button>
              )}
            </div>

            {/* Selected Tags */}
            {selectedTags.length > 0 && (
              <ChipGroup categoryName="Filtered by tags" style={{ marginBottom: '1rem' }}>
                {selectedTags.map((tag) => (
                  <Chip
                    key={tag}
                    onClick={() => setSelectedTags(selectedTags.filter((t) => t !== tag))}
                  >
                    {tag}
                  </Chip>
                ))}
              </ChipGroup>
            )}

            {/* Flow List */}
            {filteredFlows.length === 0 ? (
              <EmptyState>
                <EmptyStateIcon icon={SearchIcon} />
                <Title headingLevel="h4" size="lg">
                  No flows found
                </Title>
                <EmptyStateBody>
                  Try adjusting your search criteria or clearing filters.
                </EmptyStateBody>
                <Button variant="link" onClick={handleClearFilters}>
                  Clear filters
                </Button>
              </EmptyState>
            ) : (
              <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem' }}>
                {(() => {
                  const { sdgHub, custom } = separateFlows(filteredFlows);
                  return (
                    <>
                      {/* SDG Hub Flows */}
                      {sdgHub.length > 0 && (
                        <>
                          <div style={{ 
                            padding: '0.5rem 1rem', 
                            background: '#f0f0f0', 
                            fontWeight: 'bold',
                            fontSize: '0.875rem',
                            color: '#151515',
                            borderBottom: '2px solid #d2d2d2'
                          }}>
                            🏢 Red Hat SDG Hub Flows ({sdgHub.length})
                          </div>
                          <List isPlain isBordered>
                            {sdgHub.map((flow) => (
                              <ListItem key={flow}>
                                <div
                                  style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '0.5rem',
                                    cursor: 'pointer',
                                    borderRadius: '4px',
                                    backgroundColor:
                                      (selectedFlowDetails?.name === flow || selectedFlow?.name === flow) ? '#e7f1fa' : 'transparent',
                                  }}
                                  onClick={() => handleFlowClick(flow)}
                                >
                                  <span style={{ fontWeight: (selectedFlowDetails?.name === flow || selectedFlow?.name === flow) ? 'bold' : 'normal' }}>
                                    {flow}
                                  </span>
                                  {(selectedFlowDetails?.name === flow || selectedFlow?.name === flow) && (
                                    <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" />
                                  )}
                                </div>
                              </ListItem>
                            ))}
                          </List>
                        </>
                      )}

                      {/* Custom Flows */}
                      {custom.length > 0 && (
                        <>
                          <div style={{ 
                            padding: '0.5rem 1rem', 
                            background: '#f0f0f0', 
                            fontWeight: 'bold',
                            fontSize: '0.875rem',
                            color: '#151515',
                            borderBottom: '2px solid #d2d2d2',
                            marginTop: sdgHub.length > 0 ? '1rem' : 0
                          }}>
                            🎨 Custom Flows ({custom.length})
                          </div>
                          <List isPlain isBordered>
                            {custom.map((flow) => (
                              <ListItem key={flow}>
                                <div
                                  style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '0.5rem',
                                    cursor: 'pointer',
                                    borderRadius: '4px',
                                    backgroundColor:
                                      (selectedFlowDetails?.name === flow || selectedFlow?.name === flow) ? '#e7f1fa' : 'transparent',
                                  }}
                                  onClick={() => handleFlowClick(flow)}
                                >
                                  <span style={{ fontWeight: (selectedFlowDetails?.name === flow || selectedFlow?.name === flow) ? 'bold' : 'normal' }}>
                                    {flow}
                                  </span>
                                  {(selectedFlowDetails?.name === flow || selectedFlow?.name === flow) && (
                                    <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" />
                                  )}
                                </div>
                              </ListItem>
                            ))}
                          </List>
                        </>
                      )}
                    </>
                  );
                })()}
              </div>
            )}

            <div style={{ marginTop: 'auto', paddingTop: '1rem', fontSize: '0.875rem', color: '#6a6e73', flexShrink: 0 }}>
              <strong>{filteredFlows.length}</strong> of <strong>{flows.length}</strong> flows
            </div>
          </CardBody>
        </Card>
      </GridItem>

      {/* Right Panel - Flow Details */}
      <GridItem span={6} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Flow Details
            </Title>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            {loadingDetails ? (
              <div style={{ textAlign: 'center', padding: '2rem' }}>
                <Spinner size="lg" />
                <div style={{ marginTop: '1rem' }}>Loading flow details...</div>
              </div>
            ) : selectedFlowDetails ? (
              <div>
                <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                  {selectedFlowDetails.name}
                </Title>

                <DescriptionList isHorizontal>
                  <DescriptionListGroup>
                    <DescriptionListTerm>ID</DescriptionListTerm>
                    <DescriptionListDescription>
                      <Badge isRead>{selectedFlowDetails.id}</Badge>
                    </DescriptionListDescription>
                  </DescriptionListGroup>

                  <DescriptionListGroup>
                    <DescriptionListTerm>Version</DescriptionListTerm>
                    <DescriptionListDescription>
                      {selectedFlowDetails.version || 'N/A'}
                    </DescriptionListDescription>
                  </DescriptionListGroup>

                  <DescriptionListGroup>
                    <DescriptionListTerm>Author</DescriptionListTerm>
                    <DescriptionListDescription>
                      {selectedFlowDetails.author || 'N/A'}
                    </DescriptionListDescription>
                  </DescriptionListGroup>

                  {selectedFlowDetails.tags && selectedFlowDetails.tags.length > 0 && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Tags</DescriptionListTerm>
                      <DescriptionListDescription>
                        <ChipGroup>
                          {selectedFlowDetails.tags.map((tag) => (
                            <Chip key={tag} isReadOnly>
                              {tag}
                            </Chip>
                          ))}
                        </ChipGroup>
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}

                  {selectedFlowDetails.description && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Description</DescriptionListTerm>
                      <DescriptionListDescription>
                        {selectedFlowDetails.description}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}

                  {selectedFlowDetails.recommended_models && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Default Model</DescriptionListTerm>
                      <DescriptionListDescription>
                        <code>{selectedFlowDetails.recommended_models.default || 'N/A'}</code>
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}

                  {selectedFlowDetails.dataset_requirements && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Required Columns</DescriptionListTerm>
                      <DescriptionListDescription>
                        {selectedFlowDetails.dataset_requirements.required_columns ? (
                          <List isPlain>
                            {selectedFlowDetails.dataset_requirements.required_columns.map((col) => (
                              <ListItem key={col}>
                                <code>{col}</code>
                              </ListItem>
                            ))}
                          </List>
                        ) : (
                          'None specified'
                        )}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}
                </DescriptionList>
              </div>
            ) : (
              <EmptyState>
                <EmptyStateIcon icon={SearchIcon} />
                <Title headingLevel="h4" size="lg">
                  No flow selected
                </Title>
                <EmptyStateBody>
                  Select a flow from the list to view its details and configure it.
                </EmptyStateBody>
              </EmptyState>
            )}
          </CardBody>
        </Card>
      </GridItem>
    </Grid>
  );
};

export default FlowSelectionStep;

