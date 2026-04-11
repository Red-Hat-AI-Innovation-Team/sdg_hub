/**
 * Visual Flow Editor Constants
 * 
 * Centralized constants to avoid circular dependency issues
 * 
 * Based on actual flow.yaml patterns:
 * - LLM: PromptBuilder + LLMChat + LLMResponseExtractor (always together)
 * - Parser: TagParserBlock (extract tagged sections)
 * - Eval: LLM + Parser + Filter (evaluation with pass/fail criteria)
 * - Transform: DuplicateColumns, RenameColumns, MeltColumns
 */

import {
  CubesIcon,
  CutIcon,
  CheckCircleIcon,
  SyncAltIcon,
} from '@patternfly/react-icons';

// Node type constants (removed PROMPT - it's always part of LLM)
export const NODE_TYPES = {
  LLM: 'llm',
  PARSER: 'parser',
  EVAL: 'eval',
  TRANSFORM: 'transform',
};

// SVG icon paths for drag preview (matching PatternFly icons)
const ICON_SVGS = {
  cubes: `<svg viewBox="0 0 512 512" width="16" height="16" fill="currentColor">
    <path d="M488.6 250.2L392 214V105.5c0-15-9.3-28.4-23.4-33.7l-100-37.5c-8.1-3.1-17.1-3.1-25.3 0l-100 37.5c-14.1 5.3-23.4 18.7-23.4 33.7V214l-96.6 36.2C9.3 255.5 0 268.9 0 283.9V394c0 13.6 7.7 26.1 19.9 32.2l100 50c10.1 5.1 22.1 5.1 32.2 0l103.9-52 103.9 52c10.1 5.1 22.1 5.1 32.2 0l100-50c12.2-6.1 19.9-18.6 19.9-32.2V283.9c0-15-9.3-28.4-23.4-33.7zM358 214.8l-85 31.9v-68.2l85-37v73.3zM154 104.1l102-38.2 102 38.2v.6l-102 41.4-102-41.4v-.6zm84 291.1l-85 42.5v-79.1l85-38.8v75.4zm0-112l-102 41.4-102-41.4v-.6l102-38.2 102 38.2v.6zm240 112l-85 42.5v-79.1l85-38.8v75.4zm0-112l-102 41.4-102-41.4v-.6l102-38.2 102 38.2v.6z"/>
  </svg>`,
  cut: `<svg viewBox="0 0 448 512" width="16" height="16" fill="currentColor">
    <path d="M278.06 256L444.48 89.57c4.69-4.69 4.69-12.29 0-16.97-32.8-32.8-85.99-32.8-118.79 0L210.18 188.12l-24.86-24.86c4.31-10.92 6.68-22.81 6.68-35.26 0-53.02-42.98-96-96-96S0 74.98 0 128s42.98 96 96 96c4.54 0 8.99-.32 13.36-.93L142.29 256l-32.93 32.93c-4.37-.61-8.83-.93-13.36-.93-53.02 0-96 42.98-96 96s42.98 96 96 96 96-42.98 96-96c0-12.45-2.37-24.34-6.68-35.26l24.86-24.86L325.69 439.4c32.8 32.8 85.99 32.8 118.79 0 4.69-4.68 4.69-12.28 0-16.97L278.06 256zM96 160c-17.64 0-32-14.36-32-32s14.36-32 32-32 32 14.36 32 32-14.36 32-32 32zm0 256c-17.64 0-32-14.36-32-32s14.36-32 32-32 32 14.36 32 32-14.36 32-32 32z"/>
  </svg>`,
  checkCircle: `<svg viewBox="0 0 512 512" width="16" height="16" fill="currentColor">
    <path d="M504 256c0 136.967-111.033 248-248 248S8 392.967 8 256 119.033 8 256 8s248 111.033 248 248zM227.314 387.314l184-184c6.248-6.248 6.248-16.379 0-22.627l-22.627-22.627c-6.248-6.249-16.379-6.249-22.628 0L216 308.118l-70.059-70.059c-6.248-6.248-16.379-6.248-22.628 0l-22.627 22.627c-6.248 6.248-6.248 16.379 0 22.627l104 104c6.249 6.249 16.379 6.249 22.628.001z"/>
  </svg>`,
  syncAlt: `<svg viewBox="0 0 512 512" width="16" height="16" fill="currentColor">
    <path d="M370.72 133.28C339.458 104.008 298.888 87.962 255.848 88c-77.458.068-144.328 53.178-162.791 126.85-1.344 5.363-6.122 9.15-11.651 9.15H24.103c-7.498 0-13.194-6.807-11.807-14.176C33.933 94.924 134.813 8 256 8c66.448 0 126.791 26.136 171.315 68.685L463.03 41.97c10.937-10.938 29.629-3.188 29.629 12.284v132.284c0 9.9-8.029 17.929-17.929 17.929H342.446c-15.471 0-23.222-18.692-12.284-29.629l40.558-40.558zM32 340.464V208.18c0-9.9 8.029-17.929 17.929-17.929H182.21c15.471 0 23.222 18.692 12.284 29.629l-40.558 40.558c31.262 29.27 71.835 45.319 114.876 45.28 77.418-.07 144.315-53.144 162.787-126.849 1.344-5.363 6.122-9.15 11.651-9.15h57.304c7.498 0 13.194 6.807 11.807 14.176C490.067 417.076 389.187 504 268 504c-66.448 0-126.791-26.136-171.315-68.685L60.97 470.03c-10.937 10.938-29.629 3.188-29.629-12.284V340.464z"/>
  </svg>`,
};

// Node type configurations
export const NODE_TYPE_CONFIG = {
  [NODE_TYPES.LLM]: {
    label: 'LLM',
    icon: CubesIcon,
    iconSvg: ICON_SVGS.cubes,
    color: '#0066cc',
    description: 'Prompt + LLM call + extract response',
    shortDescription: 'LLM generation',
  },
  [NODE_TYPES.PARSER]: {
    label: 'Parser',
    icon: CutIcon,
    iconSvg: ICON_SVGS.cut,
    color: '#f0ab00',
    description: 'Extract tagged sections from text',
    shortDescription: 'Extract sections',
  },
  [NODE_TYPES.EVAL]: {
    label: 'Eval',
    icon: CheckCircleIcon,
    iconSvg: ICON_SVGS.checkCircle,
    color: '#c9190b',
    description: 'Evaluate & filter by quality criteria',
    shortDescription: 'Quality filter',
  },
  [NODE_TYPES.TRANSFORM]: {
    label: 'Transform',
    icon: SyncAltIcon,
    iconSvg: ICON_SVGS.syncAlt,
    color: '#6753ac',
    description: 'Duplicate, rename, or melt columns',
    shortDescription: 'Column ops',
  },
};

/**
 * Generate unique ID for nodes
 */
export const generateNodeId = () => `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

/**
 * Generate unique ID for edges
 */
export const generateEdgeId = () => `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
