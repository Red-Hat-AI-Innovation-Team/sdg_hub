import React, { useState, useEffect } from 'react';
import {
  PageSection,
  Title,
  Text,
  Card,
  CardBody,
  Grid,
  GridItem,
  Button,
  Flex,
  FlexItem,
  Spinner,
  Tooltip,
} from '@patternfly/react-core';
import {
  RocketIcon,
  PlusCircleIcon,
  TachometerAltIcon,
  ArrowRightIcon,
  CubesIcon,
  BoltIcon,
  DatabaseIcon,
  HomeIcon,
  ListIcon,
  HistoryIcon,
  OutlinedFileAltIcon,
  AngleRightIcon,
  AngleDownIcon,
} from '@patternfly/react-icons';
import { flowAPI } from '../services/api';

/**
 * Flow category definitions - same as FlowSelectionStep
 */
const FLOW_CATEGORIES = {
  'knowledge-generation': {
    name: 'Knowledge Generation',
    description: 'Generate Q&A pairs and knowledge datasets from documents',
    color: '#0066cc',
    order: 1,
  },
  'multilingual': {
    name: 'Multilingual',
    description: 'Generate datasets in multiple languages',
    color: '#3e8635',
    order: 2,
  },
  'skills-generation': {
    name: 'Skills Generation',
    description: 'Create skill-based training datasets',
    color: '#f0ab00',
    order: 3,
  },
  'text-analysis': {
    name: 'Text Analysis',
    description: 'Extract insights and analyze text content',
    color: '#6753ac',
    order: 4,
  },
  'evaluation': {
    name: 'Evaluation',
    description: 'Evaluate and benchmark RAG systems',
    color: '#8a8d90',
    order: 5,
  },
};

/**
 * Categorize a flow based on its tags or name - same logic as FlowSelectionStep
 */
const categorizeFlow = (flowName, flowDetails = null) => {
  const nameLower = flowName.toLowerCase();
  const tags = flowDetails?.tags || [];
  
  // Skip custom flows
  if (flowName.includes('(Custom)')) {
    return null;
  }
  
  // NAME-BASED CHECKS FIRST
  if (nameLower.includes('evaluation') || nameLower.includes('benchmark')) {
    return 'evaluation';
  }
  if (nameLower.includes('japanese') || nameLower.includes('multilingual') || nameLower.includes('korean') || nameLower.includes('chinese') || nameLower.includes('spanish') || nameLower.includes('french') || nameLower.includes('german')) {
    return 'multilingual';
  }
  
  // TAG-BASED CHECKS
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
  if (tags.includes('rag')) {
    return 'evaluation';
  }
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
  if (nameLower.includes('qa') || nameLower.includes('knowledge') || nameLower.includes('summary') || nameLower.includes('rag')) {
    return 'knowledge-generation';
  }
  
  return 'knowledge-generation'; // Default category
};

const HomeDashboard = ({ onNavigate }) => {
  const [flows, setFlows] = useState([]);
  const [flowDetailsCache, setFlowDetailsCache] = useState({});
  const [loading, setLoading] = useState(true);
  const [groupedFlows, setGroupedFlows] = useState({});
  const [expandedCategories, setExpandedCategories] = useState({ 'knowledge-generation': true });

  // Load flows from API with details in a single request (optimized)
  useEffect(() => {
    const loadFlows = async () => {
      try {
        setLoading(true);

        // Single API call gets all flows with their details
        const flowsWithDetails = await flowAPI.listFlowsWithDetails();

        // Filter out custom flows
        const sdgHubFlows = flowsWithDetails.filter(
          (flow) => !flow.name.includes('(Custom)')
        );

        // Build flows list and details cache
        const flowNames = sdgHubFlows.map((flow) => flow.name);
        const cache = {};
        sdgHubFlows.forEach((flow) => {
          cache[flow.name] = flow;
        });

        setFlows(flowNames);
        setFlowDetailsCache(cache);

        // Group flows by category
        const grouped = {};
        sdgHubFlows.forEach((flow) => {
          const category = categorizeFlow(flow.name, flow);
          if (category) {
            if (!grouped[category]) {
              grouped[category] = [];
            }
            grouped[category].push({
              name: flow.name,
              description: flow.description || '',
            });
          }
        });
        setGroupedFlows(grouped);
      } catch (error) {
        console.error('Failed to load flows:', error);
      } finally {
        setLoading(false);
      }
    };

    loadFlows();
  }, []);

  // Get sorted categories
  const sortedCategories = Object.keys(groupedFlows).sort((a, b) => {
    const orderA = FLOW_CATEGORIES[a]?.order || 50;
    const orderB = FLOW_CATEGORIES[b]?.order || 50;
    return orderA - orderB;
  });

  const toggleCategory = (categoryId) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [categoryId]: !prev[categoryId],
    }));
  };

  return (
    <PageSection>
      {/* Hero Section - Compact */}
      <Card style={{ marginBottom: '1.5rem', background: 'linear-gradient(135deg, #0066cc 0%, #004080 100%)' }}>
        <CardBody style={{ padding: '2rem 2.5rem' }}>
          <Grid hasGutter>
            <GridItem span={9}>
              <Title headingLevel="h1" size="2xl" style={{ color: 'white', marginBottom: '0.75rem' }}>
                Welcome to SDG Hub
              </Title>
              <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: '1.05rem', marginBottom: '1rem', lineHeight: 1.5 }}>
                Your comprehensive platform for <strong>Synthetic Data Generation</strong>. 
                Transform documents into high-quality training datasets for Large Language Models.
              </Text>
              
              {/* Flow Explanation */}
              <div style={{ 
                padding: '0.75rem 1rem', 
                backgroundColor: 'rgba(255,255,255,0.15)', 
                borderRadius: '6px',
                borderLeft: '3px solid rgba(255,255,255,0.5)',
                marginBottom: '1.25rem',
                maxWidth: '600px'
              }}>
                <Text style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.95)', lineHeight: 1.5 }}>
                  <strong style={{ color: 'white' }}>What is a Flow?</strong> A flow is a pipeline that processes your documents through a series of steps (called blocks) to generate synthetic training data. Each block performs a specific task like summarization, question generation, or evaluation.
                </Text>
              </div>
              
              {/* Quick Start */}
              <Text style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                Quick Start
              </Text>
              <Button 
                variant="primary" 
                icon={<PlusCircleIcon />} 
                onClick={() => onNavigate('configure-flow', {})} 
                style={{ backgroundColor: 'white', color: '#0066cc', fontWeight: 600 }}
              >
                Create New Flow
              </Button>
            </GridItem>
            <GridItem span={3} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <RocketIcon style={{ fontSize: '6rem', color: 'rgba(255,255,255,0.15)' }} />
            </GridItem>
          </Grid>
        </CardBody>
      </Card>

      {/* Main Content Grid */}
      <Grid hasGutter>
        {/* Left Column */}
        <GridItem span={5}>
          {/* UI Explanation */}
          <Card style={{ marginBottom: '1.5rem' }}>
            <CardBody style={{ padding: '1.5rem' }}>
              <Title headingLevel="h2" size="lg" style={{ marginBottom: '1rem' }}>
                <OutlinedFileAltIcon style={{ marginRight: '0.5rem', color: '#0066cc' }} />
                UI Pages Guide
              </Title>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div 
                  style={{ display: 'flex', gap: '0.75rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8', cursor: 'pointer' }}
                  onClick={() => onNavigate('home')}
                >
                  <HomeIcon style={{ fontSize: '1.25rem', color: '#0066cc', flexShrink: 0, marginTop: '0.1rem' }} />
                  <div>
                    <Text style={{ fontWeight: 600, fontSize: '0.9rem' }}>Home</Text>
                    <Text style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                      Introduction to SDG Hub, available flows, and quick navigation
                    </Text>
                  </div>
                </div>
                <div 
                  style={{ display: 'flex', gap: '0.75rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8', cursor: 'pointer' }}
                  onClick={() => onNavigate('dashboard')}
                >
                  <TachometerAltIcon style={{ fontSize: '1.25rem', color: '#3e8635', flexShrink: 0, marginTop: '0.1rem' }} />
                  <div>
                    <Text style={{ fontWeight: 600, fontSize: '0.9rem' }}>Dashboard</Text>
                    <Text style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                      Monitor your flows, view statistics, and manage preprocessed datasets
                    </Text>
                  </div>
                </div>
                <div 
                  style={{ display: 'flex', gap: '0.75rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8', cursor: 'pointer' }}
                  onClick={() => onNavigate('flows')}
                >
                  <CubesIcon style={{ fontSize: '1.25rem', color: '#f0ab00', flexShrink: 0, marginTop: '0.1rem' }} />
                  <div>
                    <Text style={{ fontWeight: 600, fontSize: '0.9rem' }}>Data Generation Flows</Text>
                    <Text style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                      Manage flow configurations, run generations, and monitor progress
                    </Text>
                  </div>
                </div>
                <div 
                  style={{ display: 'flex', gap: '0.75rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '6px', border: '1px solid #e8e8e8', cursor: 'pointer' }}
                  onClick={() => onNavigate('flow-runs')}
                >
                  <HistoryIcon style={{ fontSize: '1.25rem', color: '#8a8d90', flexShrink: 0, marginTop: '0.1rem' }} />
                  <div>
                    <Text style={{ fontWeight: 600, fontSize: '0.9rem' }}>Run History</Text>
                    <Text style={{ fontSize: '0.8rem', color: '#6a6e73' }}>
                      View past generation runs and download completed datasets
                    </Text>
                  </div>
                </div>
              </div>
            </CardBody>
          </Card>

          {/* Key Capabilities - Compact */}
          <Card>
            <CardBody style={{ padding: '1.5rem' }}>
              <Title headingLevel="h2" size="lg" style={{ marginBottom: '1rem' }}>
                <BoltIcon style={{ marginRight: '0.5rem', color: '#0066cc' }} />
                Key Capabilities
              </Title>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {[
                  { icon: <ListIcon />, text: 'Pre-built generation pipelines' },
                  { icon: <CubesIcon />, text: 'Custom flow builder' },
                  { icon: <DatabaseIcon />, text: 'PDF document processing' },
                  { icon: <OutlinedFileAltIcon />, text: 'JSONL dataset export' },
                ].map((item, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', backgroundColor: '#f9f9f9', borderRadius: '4px' }}>
                    <span style={{ color: '#0066cc' }}>{item.icon}</span>
                    <Text style={{ fontSize: '0.85rem' }}>{item.text}</Text>
                  </div>
                ))}
              </div>
            </CardBody>
          </Card>
        </GridItem>

        {/* Right Column - SDG Hub Prebuilt Flows */}
        <GridItem span={7}>
          <Card style={{ height: '100%' }}>
            <CardBody style={{ padding: '1.5rem' }}>
              <Flex justifyContent={{ default: 'justifyContentSpaceBetween' }} alignItems={{ default: 'alignItemsCenter' }} style={{ marginBottom: '1rem' }}>
                <FlexItem>
                  <Title headingLevel="h2" size="lg">
                    <CubesIcon style={{ marginRight: '0.5rem', color: '#3e8635' }} />
                    SDG Hub Prebuilt Flows
                  </Title>
                </FlexItem>
                <FlexItem>
                  <Button variant="link" onClick={() => onNavigate('configure-flow', {})}>
                    Create Flow <ArrowRightIcon style={{ marginLeft: '0.25rem' }} />
                  </Button>
                </FlexItem>
              </Flex>
              
              {loading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
                  <Spinner size="lg" />
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {sortedCategories.map((categoryId) => {
                    const category = FLOW_CATEGORIES[categoryId];
                    const categoryFlows = groupedFlows[categoryId] || [];
                    const isExpanded = !!expandedCategories[categoryId];
                    
                    if (categoryFlows.length === 0) return null;
                    
                    return (
                      <div key={categoryId}>
                        <div
                          onClick={() => toggleCategory(categoryId)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            cursor: 'pointer',
                            padding: '0.4rem 0.25rem',
                            borderRadius: '4px',
                            marginBottom: isExpanded ? '0.5rem' : 0,
                            userSelect: 'none',
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#f0f0f0'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                        >
                          {isExpanded
                            ? <AngleDownIcon style={{ fontSize: '0.9rem', color: category?.color || '#151515' }} />
                            : <AngleRightIcon style={{ fontSize: '0.9rem', color: category?.color || '#151515' }} />
                          }
                          <Text style={{ fontWeight: 600, fontSize: '0.9rem', color: category?.color || '#151515' }}>
                            {category?.name || categoryId}
                          </Text>
                          <Text style={{ fontSize: '0.75rem', color: '#6a6e73', marginLeft: '0.25rem' }}>
                            ({categoryFlows.length})
                          </Text>
                        </div>
                        {isExpanded && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingLeft: '0.5rem', borderLeft: `3px solid ${category?.color || '#0066cc'}` }}>
                            {categoryFlows.map((flow) => (
                              <Tooltip key={flow.name} content="Click to configure this flow" position="top" entryDelay={400}>
                                <div 
                                  style={{ 
                                    padding: '0.6rem 0.75rem', 
                                    backgroundColor: '#f9f9f9', 
                                    borderRadius: '4px',
                                    border: '1px solid #e8e8e8',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                  }}
                                  onClick={() => onNavigate('configure-flow', {
                                    sourceType: 'existing',
                                    preSelectedFlowName: flow.name,
                                    startStepName: 'select-existing',
                                  })}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = '#e7f1fa';
                                    e.currentTarget.style.borderColor = '#0066cc';
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = '#f9f9f9';
                                    e.currentTarget.style.borderColor = '#e8e8e8';
                                  }}
                                >
                                  <Text style={{ fontWeight: 500, fontSize: '0.85rem', marginBottom: flow.description ? '0.15rem' : 0, color: '#0066cc' }}>
                                    {flow.name}
                                  </Text>
                                  {flow.description && (
                                    <Text style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                                      {flow.description}
                                    </Text>
                                  )}
                                </div>
                              </Tooltip>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {sortedCategories.length === 0 && !loading && (
                    <Text style={{ color: '#6a6e73', fontStyle: 'italic' }}>
                      No prebuilt flows available
                    </Text>
                  )}
                </div>
              )}
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
    </PageSection>
  );
};

export default HomeDashboard;
