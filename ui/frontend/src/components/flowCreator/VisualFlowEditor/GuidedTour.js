import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Title,
  Text,
  TextContent,
  Alert,
  AlertVariant,
} from '@patternfly/react-core';
import {
  PlayIcon,
  TimesIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  CheckCircleIcon,
  CubesIcon,
  HandPointerIcon,
  InfoCircleIcon,
} from '@patternfly/react-icons';

const TOUR_STORAGE_KEY = 'sdg_hub_flow_builder_tour_completed';

/**
 * Pre-built LLM node configuration for the tour
 */
const TOUR_LLM_CONFIG = {
  block_name: 'analyze_text',
  input_cols: ['document'],
  output_cols: 'analysis',
  system_message: 'You are a helpful AI assistant specialized in text analysis. Provide clear and structured responses.',
  user_message: `Analyze the following document and extract key insights.

{{document}}

Provide your analysis in this format:
[SUMMARY]
A brief summary of the main points
[/SUMMARY]

[KEY_POINTS]
The most important takeaways
[/KEY_POINTS]`,
  temperature: 0.7,
  max_tokens: 2048,
};

/**
 * Pre-built Parser node configuration for the tour
 */
const TOUR_PARSER_CONFIG = {
  block_name: 'extract_summary',
  input_cols: 'extract_analyze_text_content',
  output_cols: ['summary', 'key_points'],
  start_tags: ['[SUMMARY]', '[KEY_POINTS]'],
  end_tags: ['[/SUMMARY]', '[/KEY_POINTS]'],
};

/**
 * Tour steps configuration
 * position: 'left' | 'right' | 'bottom' | 'top' | 'center'
 * action: 'next' | 'ok' | 'wait-for-node-add' | 'wait-for-save' | 'wait-for-connection' | 'confirm' | 'wait-for-test-complete' | 'wait-for-template-add' | 'wait-for-clear' | 'finish'
 */
const TOUR_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to the Custom Flow Builder',
    content: `Let's create your first flow together!

In this quick tour, you'll learn how to:
- Use pre-built flow templates
- Add and configure LLM nodes
- Add Parser nodes to extract structured data
- Test your flow with sample data

Ready to get started?`,
    target: null,
    position: 'center',
    action: 'next',
  },
  {
    id: 'templates-intro',
    title: 'Flow Templates',
    content: `Before building from scratch, let's see the Flow Templates!

Click on the "Flow Templates" tab to see pre-built flows available in SDG Hub.`,
    target: 'flow-templates-tab',
    highlightSelector: '[data-tour="flow-templates-tab"]',
    position: 'right',
    action: 'wait-for-tab-click',
    expectedTab: 'flow-templates',
    actionHint: 'Click on Flow Templates tab',
  },
  {
    id: 'templates-overview',
    title: 'Pre-built Flows',
    content: `Here you can see all the pre-built flows available in SDG Hub!

These templates include common patterns like:
- Knowledge generation
- Question-answer generation  
- Text summarization
- And more...

You can use these as starting points or run them directly. Now let's learn to build your own flow!`,
    target: null,
    position: 'center',
    action: 'next',
  },
  {
    id: 'return-to-node-library',
    title: 'Return to Node Library',
    content: `Click the "Node Library" tab to start building your own flow.`,
    target: 'node-library-tab',
    highlightSelector: '[data-tour="node-library-tab"]',
    position: 'right',
    action: 'wait-for-tab-click',
    expectedTab: 'node-library',
    actionHint: 'Click on Node Library tab',
  },
  {
    id: 'node-library-intro',
    title: 'Build From Scratch',
    content: `Here you'll find different types of nodes:

- LLM: Generate content using AI
- Parser: Extract structured data from text
- Eval: Evaluate and filter results
- Transform: Modify data columns

Let's start by adding an LLM node!`,
    target: null,
    position: 'center',
    action: 'ok',
    okButtonText: 'OK',
  },
  {
    id: 'node-library-llm',
    title: 'Drag the LLM Node',
    content: `Drag and drop the LLM node onto the canvas.`,
    target: 'node-library-llm',
    highlightSelector: '[data-tour="node-library-llm"]',
    position: 'left',
    action: 'wait-for-node-add',
    nodeType: 'llm',
    actionHint: 'Drag and drop the LLM block to the canvas',
  },
  {
    id: 'llm-config-intro',
    title: 'Step 2: Configure the LLM Node',
    content: `Great! We've pre-filled a sample configuration.

- Block Name: Unique identifier
- Input Variables: Dataset columns ({{document}})
- Output Column: Where results go

Let's look at the prompts!`,
    target: 'config-drawer',
    position: 'left',
    action: 'next',
  },
  {
    id: 'system-message',
    title: 'Step 3: System Message',
    content: `The System Message sets the AI's behavior and persona.

Notice the info icon - click it anytime for helpful tips!`,
    target: 'system-message-field',
    highlightSelector: '[data-tour="system-message"]',
    position: 'left',
    action: 'next',
  },
  {
    id: 'user-message',
    title: 'Step 4: User Message Template',
    content: `Your main prompt with:
- {{document}} - Input variables from your dataset
- [SUMMARY]...[/SUMMARY] - Tags for parsing later

Click the info icon for more examples!`,
    target: 'user-message-field',
    highlightSelector: '[data-tour="user-message"]',
    position: 'left',
    action: 'next',
  },
  {
    id: 'save-llm',
    title: 'Step 5: Save the LLM Node',
    content: `Click Save to add the node to your flow canvas.`,
    target: 'save-button',
    highlightSelector: '[data-tour="save-button"]',
    position: 'left',
    action: 'wait-for-save',
    actionHint: 'Click Save',
  },
  {
    id: 'add-parser-intro',
    title: 'Step 6: Add a Parser Node',
    content: `Now add a Parser to extract the tagged content from the LLM's response.

The Parser will look for your [SUMMARY] and [KEY_POINTS] tags and extract them into separate columns.`,
    target: null,
    position: 'center',
    action: 'ok',
    okButtonText: 'OK',
  },
  {
    id: 'add-parser',
    title: 'Step 6: Drag the Parser Node',
    content: `Drag and drop the Parser node onto the canvas.`,
    target: 'node-library-parser',
    highlightSelector: '[data-tour="node-library-parser"]',
    position: 'left',
    action: 'wait-for-node-add',
    nodeType: 'parser',
    actionHint: 'Drag and drop the Parser block to the canvas',
  },
  {
    id: 'parser-input-column',
    title: 'Step 7: Auto-Filled Input Column',
    content: `Notice how the Input Column was automatically filled!

When you connect blocks in your flow, the Parser automatically uses the output from the previous LLM node.

This happens automatically when you build your own flows too - no manual configuration needed.`,
    target: 'config-drawer',
    highlightSelector: '[data-tour="input-column-field"]',
    position: 'left',
    action: 'next',
  },
  {
    id: 'parser-tags',
    title: 'Step 8: Auto-Extracted Tags',
    content: `The Start and End Tags were automatically extracted from your LLM prompt template!

The flow builder detected [SUMMARY] and [KEY_POINTS] tags in your prompt and pre-filled them here.

When you create your own flows, any tags you use in your prompts will be automatically extracted for the Parser.`,
    target: 'config-drawer',
    highlightSelector: '[data-tour="start-tags-field"]',
    position: 'left',
    action: 'next',
  },
  {
    id: 'parser-save',
    title: 'Step 9: Save the Parser',
    content: `Click Save to add the Parser to your flow.`,
    target: 'save-button',
    highlightSelector: '[data-tour="save-button"]',
    position: 'top',
    action: 'wait-for-save',
    actionHint: 'Click Save',
  },
  {
    id: 'connect-nodes',
    title: 'Step 10: Connect the Nodes',
    content: `Drag from the output port (highlighted circle on the right) of the LLM node to the input port (left side) of the Parser node.

This creates the data flow between nodes.`,
    target: 'llm-output-port',
    highlightSelector: '[data-tour="llm-output-port"]',
    position: 'bottom',
    action: 'wait-for-connection',
    actionHint: 'Drag from the highlighted output port to the Parser input',
  },
  {
    id: 'test-intro',
    title: 'Step 11: Test Your Flow',
    content: `Your flow is ready! Click the Test button to open the test panel.

You'll need to provide your model configuration and sample data.`,
    target: 'test-button',
    highlightSelector: '[data-tour="test-button"]',
    position: 'bottom',
    action: 'wait-for-test-open',
    actionHint: 'Click the Test button',
  },
  {
    id: 'test-run',
    title: 'Step 12: Run the Test',
    content: `Fill in the test configuration:
- Model Name (e.g., meta-llama/Llama-3.3-70B-Instruct)
- Base URL (your API endpoint)
- API Key

Then click Run Test!

The guide will automatically continue when your test completes.`,
    target: null,
    position: 'bottom-right',
    action: 'wait-for-test-complete',
    actionHint: 'Click Run Test to continue',
  },
  {
    id: 'complete',
    title: 'Congratulations!',
    content: `You've successfully built and tested your first flow!

What you learned:
- Adding LLM and Parser nodes
- Configuring prompts with variables
- Connecting nodes together
- Testing your flow

Next steps:
- Try adding an Eval node to filter results
- Explore Flow Templates for more examples
- Save your flow and use it in the wizard`,
    target: null,
    position: 'center',
    action: 'finish',
  },
];

/**
 * Guided Tour Component
 */
const GuidedTour = ({
  isFirstVisit,
  forceStart = false,
  onStartTour,
  onSkipTour,
  onInjectConfig,
  onDeleteNode,
  onDeleteEdge,
  onCloseConfigDrawer,
  onOpenTestModal,
  nodes,
  edges,
  selectedNodeId,
  showConfigDrawer,
  isTestRunning,
  testResults,
  onTourComplete,
  onTourActiveChange,
  sidebarActiveTab,
  showTestModal,
}) => {
  const [showWelcome, setShowWelcome] = useState(false);
  const [tourActive, setTourActive] = useState(false);
  
  // Notify parent when tour active state changes
  useEffect(() => {
    onTourActiveChange?.(tourActive);
  }, [tourActive, onTourActiveChange]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [popupPosition, setPopupPosition] = useState({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
  const [arrowDirection, setArrowDirection] = useState(null);
  const [arrowOffset, setArrowOffset] = useState(null); // For precise arrow positioning
  
  // Stack to track actions for undo functionality
  const [actionsStack, setActionsStack] = useState([]);
  
  // Track nodes/edges count to detect additions
  const prevNodesCountRef = useRef(nodes.length);
  const prevEdgesCountRef = useRef(edges.length);

  const currentStep = TOUR_STEPS[currentStepIndex];
  
  // Calculate popup position based on target element
  useEffect(() => {
    if (!tourActive || !currentStep) return;
    
    const calculatePosition = () => {
      const position = currentStep.position || 'center';
      
      // Handle center position or no highlight selector
      if (position === 'center') {
        setPopupPosition({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      
      // Handle corner positions (no highlight needed)
      if (position === 'bottom-left') {
        setPopupPosition({ bottom: '100px', left: '24px', top: 'auto', transform: 'none' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      
      if (position === 'bottom-right') {
        setPopupPosition({ bottom: '100px', right: '24px', top: 'auto', left: 'auto', transform: 'none' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      
      // Handle dynamic node target (find node by block_name)
      let element = null;
      if (currentStep.dynamicNodeTarget) {
        // Find the node element by searching for the node with matching block_name
        const targetNode = nodes.find(n => n.config?.block_name === currentStep.dynamicNodeTarget);
        if (targetNode) {
          // Find the DOM element for this node (nodes have data-node-id attribute)
          element = document.querySelector(`[data-node-id="${targetNode.id}"]`);
        }
      } else if (currentStep.highlightSelector) {
        element = document.querySelector(currentStep.highlightSelector);
      }
      
      if (!element) {
        setPopupPosition({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      
      // For dynamic node targets, we don't need the additional check below
      if (!currentStep.highlightSelector && !currentStep.dynamicNodeTarget) {
        setPopupPosition({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      if (!element) {
        setPopupPosition({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
        setArrowDirection(null);
        setArrowOffset(null);
        return;
      }
      
      const rect = element.getBoundingClientRect();
      const popupWidth = 360;
      const popupHeight = 280;
      const margin = 16;
      
      let newPosition = {};
      let arrow = null;
      let offset = null;
      
      switch (position) {
        case 'right':
          newPosition = {
            top: `${Math.max(80, Math.min(rect.top, window.innerHeight - popupHeight - 80))}px`,
            left: `${rect.right + margin}px`,
            transform: 'none',
          };
          arrow = 'left';
          // Calculate vertical offset for arrow to point at element center
          const popupTopRight = Math.max(80, Math.min(rect.top, window.innerHeight - popupHeight - 80));
          const elementCenterYRight = rect.top + rect.height / 2;
          offset = Math.max(20, Math.min(elementCenterYRight - popupTopRight, popupHeight - 40));
          break;
        case 'left':
          // Ensure popup doesn't go off-screen to the left
          const leftPosition = Math.max(margin, rect.left - popupWidth - margin);
          newPosition = {
            top: `${Math.max(80, Math.min(rect.top, window.innerHeight - popupHeight - 80))}px`,
            left: `${leftPosition}px`,
            transform: 'none',
          };
          arrow = 'right';
          // Calculate vertical offset for arrow to point at element center
          const popupTopLeft = Math.max(80, Math.min(rect.top, window.innerHeight - popupHeight - 80));
          const elementCenterYLeft = rect.top + rect.height / 2;
          offset = Math.max(20, Math.min(elementCenterYLeft - popupTopLeft, popupHeight - 40));
          break;
        case 'bottom':
          newPosition = {
            top: `${rect.bottom + margin}px`,
            left: `${Math.max(margin, Math.min(rect.left + rect.width / 2 - popupWidth / 2, window.innerWidth - popupWidth - margin))}px`,
            transform: 'none',
          };
          arrow = 'top';
          // Calculate horizontal offset for arrow to point at element center
          const popupLeftBottom = Math.max(margin, Math.min(rect.left + rect.width / 2 - popupWidth / 2, window.innerWidth - popupWidth - margin));
          const elementCenterXBottom = rect.left + rect.width / 2;
          offset = Math.max(20, Math.min(elementCenterXBottom - popupLeftBottom, popupWidth - 40));
          break;
        case 'top':
          newPosition = {
            top: `${rect.top - popupHeight - margin}px`,
            left: `${Math.max(margin, Math.min(rect.left + rect.width / 2 - popupWidth / 2, window.innerWidth - popupWidth - margin))}px`,
            transform: 'none',
          };
          arrow = 'bottom';
          // Calculate horizontal offset for arrow to point at element center
          const popupLeftTop = Math.max(margin, Math.min(rect.left + rect.width / 2 - popupWidth / 2, window.innerWidth - popupWidth - margin));
          const elementCenterXTop = rect.left + rect.width / 2;
          offset = Math.max(20, Math.min(elementCenterXTop - popupLeftTop, popupWidth - 40));
          break;
        default:
          newPosition = { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
      }
      
      setPopupPosition(newPosition);
      setArrowDirection(arrow);
      setArrowOffset(offset);
    };
    
    calculatePosition();
    
    // Recalculate on resize/scroll
    const handleUpdate = () => requestAnimationFrame(calculatePosition);
    window.addEventListener('resize', handleUpdate);
    window.addEventListener('scroll', handleUpdate, true);
    
    // Also recalculate periodically for dynamic content
    const interval = setInterval(calculatePosition, 500);
    
    return () => {
      window.removeEventListener('resize', handleUpdate);
      window.removeEventListener('scroll', handleUpdate, true);
      clearInterval(interval);
    };
  }, [tourActive, currentStepIndex, currentStep]);

  // Check if this is the first visit
  useEffect(() => {
    if (isFirstVisit) {
      const tourCompleted = localStorage.getItem(TOUR_STORAGE_KEY);
      if (!tourCompleted) {
        setShowWelcome(true);
      }
    }
  }, [isFirstVisit]);

  // Handle manual force start of tour
  useEffect(() => {
    if (forceStart) {
      setShowWelcome(true);
      setTourActive(false);
      setCurrentStepIndex(0);
      setActionsStack([]);
    }
  }, [forceStart]);

  // Watch for node additions during tour - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-node-add') return;

    const expectedNodeType = step.nodeType;
    const newNode = nodes.find(n => n.type === expectedNodeType && !n.configured);
    
    if (newNode && nodes.length > prevNodesCountRef.current) {
      // Inject the pre-built config
      const config = expectedNodeType === 'llm' ? TOUR_LLM_CONFIG : TOUR_PARSER_CONFIG;
      onInjectConfig?.(newNode.id, config);
      
      // Record action for undo
      setActionsStack(prev => [...prev, {
        stepId: step.id,
        type: 'node-add',
        data: { nodeId: newNode.id, nodeType: expectedNodeType }
      }]);
      
      // Auto-advance after small delay
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
    
    prevNodesCountRef.current = nodes.length;
  }, [nodes, tourActive, currentStepIndex, onInjectConfig]);

  // Watch for save action - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-save') return;

    // If config drawer closed after being open, save was clicked
    if (!showConfigDrawer && selectedNodeId) {
      // Record action for undo
      const savedNode = nodes.find(n => n.id === selectedNodeId);
      if (savedNode) {
        setActionsStack(prev => [...prev, {
          stepId: step.id,
          type: 'save',
          data: { nodeId: selectedNodeId, nodeType: savedNode.type }
        }]);
      }
      
      // Auto-advance
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
  }, [showConfigDrawer, tourActive, currentStepIndex, selectedNodeId, nodes]);

  // Watch for connection - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-connection') return;

    if (edges.length > prevEdgesCountRef.current) {
      // Find the new edge
      const newEdge = edges[edges.length - 1];
      
      // Record action for undo
      setActionsStack(prev => [...prev, {
        stepId: step.id,
        type: 'edge',
        data: { edgeId: newEdge?.id, edge: newEdge }
      }]);
      
      // Auto-advance
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
    
    prevEdgesCountRef.current = edges.length;
  }, [edges, tourActive, currentStepIndex]);

  // Watch for template addition - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-template-add') return;

    // Template adds multiple nodes at once (more than 1)
    if (nodes.length > prevNodesCountRef.current && nodes.length >= 1) {
      // Auto-advance after template is loaded
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 500);
    }
    
    prevNodesCountRef.current = nodes.length;
  }, [nodes, tourActive, currentStepIndex]);

  // Watch for tab click - AUTO-ADVANCE
  const prevTabRef = useRef(sidebarActiveTab);
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-tab-click') return;

    // Check if user clicked the expected tab
    if (sidebarActiveTab === step.expectedTab && sidebarActiveTab !== prevTabRef.current) {
      // Auto-advance after tab click
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
    
    prevTabRef.current = sidebarActiveTab;
  }, [sidebarActiveTab, tourActive, currentStepIndex]);

  // Watch for test modal open - AUTO-ADVANCE
  const prevTestModalRef = useRef(showTestModal);
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-test-open') return;

    // Check if test modal just opened
    if (showTestModal && !prevTestModalRef.current) {
      // Auto-advance after test modal opens
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
    
    prevTestModalRef.current = showTestModal;
  }, [showTestModal, tourActive, currentStepIndex]);

  // Watch for specific node click - AUTO-ADVANCE
  const prevSelectedNodeRef = useRef(selectedNodeId);
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-node-click') return;

    // Check if a node was just selected (changed from previous)
    if (selectedNodeId && selectedNodeId !== prevSelectedNodeRef.current) {
      const selectedNode = nodes.find(n => n.id === selectedNodeId);
      
      if (selectedNode) {
        let isMatch = false;
        
        // Check by expected node name (block_name in config)
        if (step.expectedNodeName) {
          isMatch = selectedNode.config?.block_name === step.expectedNodeName;
        }
        // Check by expected node type
        else if (step.expectedNodeType) {
          isMatch = selectedNode.type === step.expectedNodeType;
        }
        
        if (isMatch) {
          // Auto-advance after small delay
          setTimeout(() => {
            setCurrentStepIndex(prev => prev + 1);
          }, 300);
        }
      }
    }
    
    prevSelectedNodeRef.current = selectedNodeId;
  }, [selectedNodeId, nodes, tourActive, currentStepIndex]);

  // Watch for clear all action - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-clear') return;

    // Check if all nodes were cleared
    if (nodes.length === 0 && prevNodesCountRef.current > 0) {
      // Auto-advance after clear
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
    
    prevNodesCountRef.current = nodes.length;
  }, [nodes, tourActive, currentStepIndex]);

  // Watch for test completion - AUTO-ADVANCE
  useEffect(() => {
    if (!tourActive) return;
    
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action !== 'wait-for-test-complete') return;

    // Check if test has completed with results
    const hasResults = testResults && Object.keys(testResults).length > 0;
    if (hasResults && !isTestRunning) {
      // Auto-advance to congratulations
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 500);
    }
  }, [testResults, isTestRunning, tourActive, currentStepIndex]);

  // Handle confirm action (open test modal)
  const handleConfirmAction = useCallback(() => {
    const step = TOUR_STEPS[currentStepIndex];
    if (step?.action === 'confirm') {
      onOpenTestModal?.();
      // Advance to next step
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, 300);
    }
  }, [currentStepIndex, onOpenTestModal]);

  const handleStartTour = useCallback(() => {
    setShowWelcome(false);
    setTourActive(true);
    setCurrentStepIndex(0);
    setActionsStack([]);
    prevNodesCountRef.current = nodes.length;
    prevEdgesCountRef.current = edges.length;
    onStartTour?.();
  }, [onStartTour, nodes.length, edges.length]);

  const handleSkipTour = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem(TOUR_STORAGE_KEY, 'true');
    onSkipTour?.();
  }, [onSkipTour]);

  const handleNext = useCallback(() => {
    const step = TOUR_STEPS[currentStepIndex];
    
    if (step.action === 'finish') {
      setTourActive(false);
      localStorage.setItem(TOUR_STORAGE_KEY, 'true');
      onTourComplete?.();
      return;
    }

    // For wait-for actions, just wait (auto-advance will handle it)
    if (step.action.startsWith('wait-for')) {
      return;
    }

    setCurrentStepIndex(prev => Math.min(prev + 1, TOUR_STEPS.length - 1));
  }, [currentStepIndex, onTourComplete]);

  // Handle OK button click (for instruction steps before wait-for actions)
  const handleOkClick = useCallback(() => {
    setCurrentStepIndex(prev => Math.min(prev + 1, TOUR_STEPS.length - 1));
  }, []);

  const handleBack = useCallback(() => {
    if (currentStepIndex === 0) return;
    
    const previousStepIndex = currentStepIndex - 1;
    const previousStep = TOUR_STEPS[previousStepIndex];
    
    // Find actions from the step we're leaving and the previous step
    const currentStepActions = actionsStack.filter(a => a.stepId === TOUR_STEPS[currentStepIndex]?.id);
    const prevStepActions = actionsStack.filter(a => a.stepId === previousStep?.id);
    const actionsToUndo = [...currentStepActions, ...prevStepActions];
    
    // Undo actions in reverse order
    for (const action of actionsToUndo.reverse()) {
      if (action.type === 'node-add' && onDeleteNode) {
        onDeleteNode(action.data.nodeId);
        prevNodesCountRef.current = Math.max(0, prevNodesCountRef.current - 1);
      } else if (action.type === 'edge' && onDeleteEdge) {
        onDeleteEdge(action.data.edgeId);
        prevEdgesCountRef.current = Math.max(0, prevEdgesCountRef.current - 1);
      }
    }
    
    // Remove undone actions from stack
    setActionsStack(prev => prev.filter(a => 
      a.stepId !== TOUR_STEPS[currentStepIndex]?.id && 
      a.stepId !== previousStep?.id
    ));
    
    // Close config drawer if open
    if (showConfigDrawer && onCloseConfigDrawer) {
      onCloseConfigDrawer();
    }
    
    setCurrentStepIndex(previousStepIndex);
  }, [currentStepIndex, actionsStack, onDeleteNode, onDeleteEdge, onCloseConfigDrawer, showConfigDrawer]);

  const handleEndTour = useCallback(() => {
    setTourActive(false);
    localStorage.setItem(TOUR_STORAGE_KEY, 'true');
    onTourComplete?.();
  }, [onTourComplete]);

  // Welcome Modal
  if (showWelcome) {
    return (
      <Modal
        variant={ModalVariant.small}
        title=""
        isOpen={true}
        showClose={false}
        aria-labelledby="welcome-modal-title"
        style={{ '--pf-v5-c-modal-box--MaxWidth': '480px' }}
      >
        <div style={{ textAlign: 'center', padding: '1rem' }}>
          <div style={{
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #0066cc 0%, #004080 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.5rem auto',
          }}>
            <CubesIcon style={{ fontSize: '36px', color: 'white' }} />
          </div>
          
          <Title headingLevel="h2" size="xl" style={{ marginBottom: '1rem' }}>
            Welcome to the Custom Flow Builder
          </Title>
          
          <TextContent style={{ marginBottom: '2rem' }}>
            <Text>
              Build powerful AI workflows by connecting nodes together. 
              Create prompts, process responses, and automate complex tasks.
            </Text>
          </TextContent>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <Button
              variant="primary"
              size="lg"
              icon={<PlayIcon />}
              onClick={handleStartTour}
              isBlock
            >
              Start Interactive Tour
            </Button>
            <Button
              variant="secondary"
              onClick={handleSkipTour}
              isBlock
            >
              Skip and Start Building
            </Button>
          </div>

          <Text component="small" style={{ marginTop: '1.5rem', color: '#6a6e73' }}>
            The tour takes about 2 minutes and will guide you through creating your first flow.
          </Text>
        </div>
      </Modal>
    );
  }

  // Tour Step Overlay
  if (tourActive && currentStep) {
    const isWaitForAction = currentStep.action?.startsWith('wait-for');
    const isConfirmAction = currentStep.action === 'confirm';
    const isOkAction = currentStep.action === 'ok';
    const showBackdrop = currentStep.position === 'center';
    
    // Hide popup when test is running (so user can see results clearly)
    const isTestRelatedStep = currentStep.id === 'test-run' || currentStep.action === 'wait-for-test-complete';
    if (isTestRelatedStep && isTestRunning) {
      // Don't show popup while test is running - let user see results
      return null;
    }

    // Arrow styles based on direction - positioned to point at target element
    const arrowStyles = arrowDirection ? {
      position: 'absolute',
      width: 0,
      height: 0,
      ...(arrowDirection === 'left' && {
        left: '-12px',
        top: arrowOffset ? `${arrowOffset}px` : '50%',
        transform: arrowOffset ? 'translateY(-50%)' : 'translateY(-50%)',
        borderTop: '12px solid transparent',
        borderBottom: '12px solid transparent',
        borderRight: '12px solid white',
      }),
      ...(arrowDirection === 'right' && {
        right: '-12px',
        top: arrowOffset ? `${arrowOffset}px` : '50%',
        transform: arrowOffset ? 'translateY(-50%)' : 'translateY(-50%)',
        borderTop: '12px solid transparent',
        borderBottom: '12px solid transparent',
        borderLeft: '12px solid white',
      }),
      ...(arrowDirection === 'top' && {
        top: '-12px',
        left: arrowOffset ? `${arrowOffset}px` : '50%',
        transform: arrowOffset ? 'translateX(-50%)' : 'translateX(-50%)',
        borderLeft: '12px solid transparent',
        borderRight: '12px solid transparent',
        borderBottom: '12px solid white',
      }),
      ...(arrowDirection === 'bottom' && {
        bottom: '-12px',
        left: arrowOffset ? `${arrowOffset}px` : '50%',
        transform: arrowOffset ? 'translateX(-50%)' : 'translateX(-50%)',
        borderLeft: '12px solid transparent',
        borderRight: '12px solid transparent',
        borderTop: '12px solid white',
      }),
    } : null;

    return (
      <>
        {/* Semi-transparent backdrop for center/modal-style steps */}
        {showBackdrop && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 9998,
          }} />
        )}

        {/* Tour Step Card - dynamically positioned */}
        <div style={{
          position: 'fixed',
          ...popupPosition,
          width: '360px',
          background: 'white',
          borderRadius: '8px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.25)',
          zIndex: 9999,
          overflow: 'visible',
        }}>
          {/* Arrow pointer */}
          {arrowStyles && <div style={arrowStyles} />}
          
          {/* Progress Bar */}
          <div style={{
            height: '4px',
            background: '#e0e0e0',
            borderRadius: '8px 8px 0 0',
          }}>
            <div style={{
              height: '100%',
              width: `${((currentStepIndex + 1) / TOUR_STEPS.length) * 100}%`,
              background: '#0066cc',
              borderRadius: '8px 8px 0 0',
              transition: 'width 0.3s ease',
            }} />
          </div>

          {/* Content */}
          <div style={{ padding: '1rem 1.25rem' }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'flex-start',
              marginBottom: '0.75rem',
            }}>
              <Title headingLevel="h4" size="md" style={{ fontSize: '16px', fontWeight: 600 }}>
                {currentStep.title}
              </Title>
              <Button
                variant="plain"
                aria-label="Close tour"
                onClick={handleEndTour}
                style={{ padding: '2px', marginTop: '-4px' }}
              >
                <TimesIcon />
              </Button>
            </div>

            <TextContent>
              <Text style={{ 
                whiteSpace: 'pre-line',
                lineHeight: '1.5',
                fontSize: '14px',
                color: '#151515',
              }}>
                {currentStep.content.split(/\*\*(.*?)\*\*/).map((part, i) => 
                  i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                )}
              </Text>
            </TextContent>

            {/* Action hint for wait-for actions */}
            {isWaitForAction && currentStep.actionHint && (
              <div style={{
                marginTop: '0.75rem',
                padding: '0.5rem 0.75rem',
                background: '#e7f1fa',
                borderRadius: '4px',
                borderLeft: '3px solid #0066cc',
                fontSize: '13px',
                color: '#002952',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <HandPointerIcon style={{ flexShrink: 0 }} />
                <strong>{currentStep.actionHint}</strong>
              </div>
            )}

            {isWaitForAction && !currentStep.actionHint && (
              <Alert
                variant={AlertVariant.info}
                isInline
                title="Complete the action to continue"
                style={{ marginTop: '0.75rem' }}
              />
            )}
          </div>

          {/* Footer */}
          <div style={{
            padding: '0.75rem 1rem',
            background: '#f5f5f5',
            borderTop: '1px solid #e0e0e0',
            borderRadius: '0 0 8px 8px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            <Text component="small" style={{ color: '#6a6e73', fontSize: '12px' }}>
              {currentStepIndex + 1} / {TOUR_STEPS.length}
            </Text>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {currentStepIndex > 0 && (
                <Button
                  variant="link"
                  icon={<ArrowLeftIcon />}
                  onClick={handleBack}
                  size="sm"
                  style={{ padding: '4px 8px' }}
                >
                  Back
                </Button>
              )}
              
              {/* OK button for instruction steps before wait-for actions */}
              {isOkAction && (
                <Button
                  variant="primary"
                  icon={<CheckCircleIcon />}
                  onClick={handleOkClick}
                  size="sm"
                >
                  {currentStep.okButtonText || 'OK'}
                </Button>
              )}
              
              {/* Confirm button for confirm action type */}
              {isConfirmAction && (
                <Button
                  variant="primary"
                  icon={<PlayIcon />}
                  onClick={handleConfirmAction}
                  size="sm"
                >
                  {currentStep.confirmButtonText || 'Confirm'}
                </Button>
              )}
              
              {/* Show Next only for non-wait-for, non-confirm, and non-ok steps */}
              {!isWaitForAction && !isConfirmAction && !isOkAction && (
                <Button
                  variant="primary"
                  icon={currentStep.action === 'finish' ? <CheckCircleIcon /> : <ArrowRightIcon />}
                  iconPosition="right"
                  onClick={handleNext}
                  size="sm"
                >
                  {currentStep.action === 'finish' ? 'Finish' : 'Next'}
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Highlight overlay for specific elements */}
        {currentStep.highlightSelector && (
          <HighlightOverlay selector={currentStep.highlightSelector} />
        )}
        
        {/* Highlight overlay for dynamic node targets */}
        {currentStep.dynamicNodeTarget && !currentStep.highlightSelector && (
          <HighlightOverlay 
            selector={null} 
            nodeId={nodes.find(n => n.config?.block_name === currentStep.dynamicNodeTarget)?.id}
          />
        )}
      </>
    );
  }

  return null;
};

/**
 * Highlight overlay component that creates a "spotlight" effect
 * Follows the element position even when scrolling
 * @param {string} selector - CSS selector to find the element
 * @param {string} nodeId - Optional node ID to find the element by data-node-id attribute
 */
const HighlightOverlay = ({ selector, nodeId }) => {
  const [rect, setRect] = useState(null);
  const rafRef = useRef(null);

  useEffect(() => {
    let element = null;
    if (selector) {
      element = document.querySelector(selector);
    } else if (nodeId) {
      element = document.querySelector(`[data-node-id="${nodeId}"]`);
    }
    if (!element) return;

    const updateRect = () => {
      const r = element.getBoundingClientRect();
      setRect({
        top: r.top - 6,
        left: r.left - 6,
        width: r.width + 12,
        height: r.height + 12,
      });
    };
    
    // Throttled update using requestAnimationFrame
    const handleScrollOrResize = () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      rafRef.current = requestAnimationFrame(updateRect);
    };

    // Initial position
    updateRect();

    // Listen to resize and scroll events
    window.addEventListener('resize', handleScrollOrResize);
    window.addEventListener('scroll', handleScrollOrResize, true);
    
    // Also observe DOM changes
    const observer = new MutationObserver(handleScrollOrResize);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });

    return () => {
      window.removeEventListener('resize', handleScrollOrResize);
      window.removeEventListener('scroll', handleScrollOrResize, true);
      observer.disconnect();
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [selector, nodeId]);

  if (!rect) return null;

  return (
    <div style={{
      position: 'fixed',
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      border: '3px solid #0066cc',
      borderRadius: '6px',
      boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5)',
      pointerEvents: 'none',
      zIndex: 9997,
      animation: 'tourPulse 2s infinite',
    }}>
      <style>{`
        @keyframes tourPulse {
          0%, 100% { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5), 0 0 0 0 rgba(0, 102, 204, 0.4); }
          50% { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5), 0 0 0 8px rgba(0, 102, 204, 0); }
        }
      `}</style>
    </div>
  );
};

export default GuidedTour;
export { TOUR_LLM_CONFIG, TOUR_PARSER_CONFIG, TOUR_STORAGE_KEY };
