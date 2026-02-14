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
import { 
  SearchIcon, 
  CheckCircleIcon, 
  UploadIcon,
  BookOpenIcon,
  GlobeIcon,
  BullseyeIcon,
  ChartBarIcon,
  PaintBrushIcon,
} from '@patternfly/react-icons';
import { flowAPI, configAPI } from '../../services/api';

/**
 * Flow category definitions
 * Maps category IDs to display names and descriptions
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
 * Categorize a flow based on its tags or name
 * Order matters! More specific checks come first.
 * Name-based checks take priority for disambiguation.
 */
const categorizeFlow = (flowName, flowDetails = null) => {
  const nameLower = flowName.toLowerCase();
  const tags = flowDetails?.tags || [];
  
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
 * Flow Selection Step Component
 * 
 * Allows users to:
 * - Browse all available flows
 * - Search flows by tag
 * - View flow details
 * - Select a flow for configuration
 */
const FlowSelectionStep = ({ selectedFlow, onFlowSelect, onError, isImported, preSelectedFlowName }) => {
  const [flows, setFlows] = useState([]);
  const [filteredFlows, setFilteredFlows] = useState([]);
  const [selectedFlowDetails, setSelectedFlowDetails] = useState(null);
  const [searchValue, setSearchValue] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [isTagSelectOpen, setIsTagSelectOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Cache for flow details (to show descriptions without clicking)
  const [flowDetailsCache, setFlowDetailsCache] = useState({});
  const [loadingDescriptions, setLoadingDescriptions] = useState(false);

  // Expanded categories state - all collapsed by default
  const [expandedCategories, setExpandedCategories] = useState({
    'knowledge-generation': false,
    'multilingual': false,
    'skills-generation': false,
    'text-analysis': false,
    'evaluation': false,
    'custom': false,
  });

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
   * Auto-select a pre-selected flow (e.g., when navigating from Home page)
   */
  useEffect(() => {
    if (preSelectedFlowName && flows.length > 0 && !selectedFlow) {
      // Check if the flow exists in our list
      const flowExists = flows.includes(preSelectedFlowName);
      if (flowExists) {
        // Auto-expand the category this flow belongs to
        const details = flowDetailsCache[preSelectedFlowName];
        const category = categorizeFlow(preSelectedFlowName, details);
        if (category) {
          setExpandedCategories(prev => ({ ...prev, [category]: true }));
        }
        // Auto-click the flow to select it and load details
        handleFlowClick(preSelectedFlowName);
      }
    }
  }, [preSelectedFlowName, flows, flowDetailsCache]);

  /**
   * Auto-expand the category of an already-selected flow (e.g., clone or edit)
   * so the highlighted flow is visible without manual expansion
   */
  useEffect(() => {
    if (selectedFlow?.name && flows.length > 0 && Object.keys(flowDetailsCache).length > 0) {
      // Strip "(Copy)" suffix to find the base flow name in the library
      const baseName = selectedFlow.name.replace(/\s*\(Copy\)$/g, '');
      const flowExists = flows.includes(baseName) || flows.includes(selectedFlow.name);
      const matchedName = flows.includes(selectedFlow.name) ? selectedFlow.name : baseName;
      if (flowExists) {
        const details = flowDetailsCache[matchedName];
        const category = categorizeFlow(matchedName, details);
        if (category) {
          setExpandedCategories(prev => ({ ...prev, [category]: true }));
        }
      }
    }
  }, [selectedFlow, flows, flowDetailsCache]);

  // Flow descriptions are now loaded in a single request via loadFlows()
  // using the listFlowsWithDetails endpoint - no need for separate batched calls

  /**
   * Group flows by category
   */
  const groupFlowsByCategory = (flowList) => {
    const grouped = {};
    
    flowList.forEach(flowName => {
      const details = flowDetailsCache[flowName];
      const category = categorizeFlow(flowName, details);
      
      if (!grouped[category]) {
        grouped[category] = [];
      }
      grouped[category].push({
        name: flowName,
        details: details,
      });
    });
    
    // Sort categories by order
    const sortedCategories = Object.keys(grouped).sort((a, b) => {
      const orderA = FLOW_CATEGORIES[a]?.order || 50;
      const orderB = FLOW_CATEGORIES[b]?.order || 50;
      return orderA - orderB;
    });
    
    return { grouped, sortedCategories };
  };

  /**
   * Load flows from API with full details in a single request.
   * This is much faster than the old approach of listFlows + N x getFlowInfo calls.
   */
  const loadFlows = async () => {
    try {
      setLoading(true);
      setLoadingDescriptions(true);

      // Single API call gets all flows with their details
      const flowsWithDetails = await flowAPI.listFlowsWithDetails();

      // Extract flow names and build details cache
      const flowNames = flowsWithDetails.map((flow) => flow.name);
      const cache = {};
      flowsWithDetails.forEach((flow) => {
        cache[flow.name] = flow;
      });

      setFlows(flowNames);
      setFilteredFlows(flowNames);
      setFlowDetailsCache(cache);
    } catch (error) {
      onError('Failed to load flows: ' + error.message);
    } finally {
      setLoading(false);
      setLoadingDescriptions(false);
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

            {/* Flow List - Categorized */}
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
                  const { grouped, sortedCategories } = groupFlowsByCategory(filteredFlows);
                  
                  return sortedCategories.map(categoryId => {
                    const category = FLOW_CATEGORIES[categoryId] || { name: categoryId, icon: BookOpenIcon, color: '#6a6e73' };
                    const categoryFlows = grouped[categoryId] || [];
                    const isExpanded = expandedCategories[categoryId] !== false;
                    const CategoryIcon = category.icon;
                    
                    return (
                      <div key={categoryId} style={{ marginBottom: '0.5rem' }}>
                        {/* Category Header */}
                        <div
                          style={{
                            padding: '0.75rem 1rem',
                            background: '#f0f0f0',
                            fontWeight: 'bold',
                            fontSize: '0.9rem',
                            color: '#151515',
                            borderBottom: '2px solid #d2d2d2',
                            cursor: 'pointer',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                          }}
                          onClick={() => setExpandedCategories(prev => ({
                            ...prev,
                            [categoryId]: !prev[categoryId]
                          }))}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <CategoryIcon style={{ color: category.color }} />
                            <span>{category.name} ({categoryFlows.length})</span>
                          </span>
                          <span style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                            {isExpanded ? '▼' : '▶'}
                          </span>
                        </div>
                        
                        {/* Category Description */}
                        {isExpanded && (
                          <div style={{
                            padding: '0.5rem 1rem',
                            fontSize: '0.8rem',
                            color: '#6a6e73',
                            backgroundColor: '#fafafa',
                            borderBottom: '1px solid #e0e0e0',
                          }}>
                            {category.description}
                          </div>
                        )}
                        
                        {/* Flows in Category */}
                        {isExpanded && (
                          <List isPlain isBordered>
                            {categoryFlows.map(({ name: flowName, details }) => {
                              // Match by exact name, or strip clone suffix "(Copy)" to match the underlying flow
                              const selectedName = selectedFlow?.name || '';
                              const baseSelectedName = selectedName.replace(/\s*\(Copy\)$/g, '');
                              const isSelected = selectedFlowDetails?.name === flowName || selectedName === flowName || baseSelectedName === flowName;
                              const description = details?.description || 'No description available';
                              
                              return (
                                <ListItem key={flowName}>
                                  <div
                                    style={{
                                      padding: '0.75rem 1rem',
                                      cursor: 'pointer',
                                      borderRadius: '4px',
                                      backgroundColor: isSelected ? '#e7f1fa' : 'transparent',
                                      borderLeft: isSelected ? '3px solid #0066cc' : '3px solid transparent',
                                    }}
                                    onClick={() => handleFlowClick(flowName)}
                                  >
                                    {/* Flow Name */}
                                    <div style={{
                                      display: 'flex',
                                      justifyContent: 'space-between',
                                      alignItems: 'center',
                                      marginBottom: '0.25rem',
                                    }}>
                                      <span style={{
                                        fontWeight: isSelected ? 'bold' : '500',
                                        color: '#151515',
                                      }}>
                                        {flowName.replace(' (Custom)', '')}
                                        {flowName.includes('(Custom)') && (
                                          <Badge style={{ marginLeft: '0.5rem' }} isRead>Custom</Badge>
                                        )}
                                      </span>
                                      {isSelected && (
                                        <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" />
                                      )}
                                    </div>
                                    
                                    {/* Flow Description - Full text with line wrapping */}
                                    <div style={{
                                      fontSize: '0.8rem',
                                      color: '#6a6e73',
                                      lineHeight: '1.5',
                                      maxWidth: '600px',
                                    }}>
                                      {loadingDescriptions ? (
                                        <span style={{ fontStyle: 'italic' }}>Loading description...</span>
                                      ) : (
                                        description
                                      )}
                                    </div>
                                    
                                    {/* Tags - if available */}
                                    {details?.tags && details.tags.length > 0 && (
                                      <div style={{ marginTop: '0.5rem' }}>
                                        {details.tags.slice(0, 3).map(tag => (
                                          <Badge
                                            key={tag}
                                            isRead
                                            style={{
                                              marginRight: '0.25rem',
                                              fontSize: '0.7rem',
                                              backgroundColor: '#f0f0f0',
                                            }}
                                          >
                                            {tag}
                                          </Badge>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </ListItem>
                              );
                            })}
                          </List>
                        )}
                      </div>
                    );
                  });
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

                  {selectedFlowDetails.recommended_models?.compatible && 
                   selectedFlowDetails.recommended_models.compatible.length > 0 && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Compatible Models</DescriptionListTerm>
                      <DescriptionListDescription>
                        <List isPlain>
                          {selectedFlowDetails.recommended_models.compatible.map((model) => (
                            <ListItem key={model}>
                              <code>{model}</code>
                            </ListItem>
                          ))}
                        </List>
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

