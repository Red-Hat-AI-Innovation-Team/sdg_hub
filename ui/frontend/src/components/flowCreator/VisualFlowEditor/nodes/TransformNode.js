import React from 'react';
import { SyncAltIcon } from '@patternfly/react-icons';

/**
 * Transform Node Configuration Component
 * Used in the NodeConfigDrawer for configuring transform nodes
 * 
 * Transform nodes handle data manipulation operations:
 * - Duplicate Columns
 * - Rename Columns  
 * - Melt Columns (unpivot)
 */
export const TransformNodeConfig = {
  type: 'transform',
  label: 'Transform',
  icon: SyncAltIcon,
  color: '#6753ac',
  description: 'Data manipulation operations (duplicate, rename, melt columns)',
  
  // Default configuration values
  defaultConfig: {
    block_name: '',
    transform_type: 'duplicate', // duplicate, rename, melt
    // For duplicate/rename: { source: dest, ... }
    column_mapping: {},
    // For melt: arrays
    melt_input_cols: [],
    melt_output_cols: ['value', 'type'],
  },

  // Configuration fields definition
  fields: [
    {
      name: 'block_name',
      label: 'Block Name',
      type: 'text',
      required: true,
      placeholder: 'e.g., duplicate_document',
      helperText: 'A unique identifier for this transform block',
    },
    {
      name: 'transform_type',
      label: 'Transform Type',
      type: 'select',
      required: true,
      options: [
        { value: 'duplicate', label: 'Duplicate Columns' },
        { value: 'rename', label: 'Rename Columns' },
        { value: 'melt', label: 'Melt Columns (Unpivot)' },
      ],
      defaultValue: 'duplicate',
      helperText: 'Type of data transformation to perform',
    },
    // Duplicate/Rename specific fields
    {
      name: 'column_mapping',
      label: 'Column Mapping',
      type: 'keyvalue',
      required: true,
      showWhen: (config) => config.transform_type === 'duplicate' || config.transform_type === 'rename',
      keyPlaceholder: 'Source column',
      valuePlaceholder: 'Target column',
      helperText: 'Map source columns to target names',
    },
    // Melt specific fields
    {
      name: 'melt_input_cols',
      label: 'Columns to Melt',
      type: 'tags',
      required: true,
      showWhen: (config) => config.transform_type === 'melt',
      placeholder: 'Add column...',
      helperText: 'Columns to unpivot into rows',
    },
    {
      name: 'melt_output_cols',
      label: 'Output Column Names',
      type: 'tags',
      required: true,
      showWhen: (config) => config.transform_type === 'melt',
      placeholder: 'Add output name...',
      helperText: 'Names for value and type columns (default: value, type)',
    },
  ],

  // Validation function
  validate: (config) => {
    const errors = {};
    if (!config.block_name?.trim()) {
      errors.block_name = 'Block name is required';
    }
    if (!config.transform_type) {
      errors.transform_type = 'Transform type is required';
    }
    
    if (config.transform_type === 'duplicate' || config.transform_type === 'rename') {
      if (!config.column_mapping || Object.keys(config.column_mapping).length === 0) {
        errors.column_mapping = 'At least one column mapping is required';
      }
    }
    
    if (config.transform_type === 'melt') {
      if (!config.melt_input_cols || config.melt_input_cols.length === 0) {
        errors.melt_input_cols = 'At least one column to melt is required';
      }
      if (!config.melt_output_cols || config.melt_output_cols.length < 2) {
        errors.melt_output_cols = 'Two output column names are required';
      }
    }
    
    return errors;
  },

  // Generate block configuration for serialization
  toBlockConfig: (config) => {
    // Handle both `column_mapping` (from TransformNodeConfig fields) and `input_cols` (from VisualFlowEditor defaults)
    // This ensures compatibility when loading flows from templates that use input_cols
    const getColumnMapping = () => {
      if (config.column_mapping && Object.keys(config.column_mapping).length > 0) {
        return config.column_mapping;
      }
      // Fallback to input_cols if column_mapping is not set (for flows loaded from templates)
      if (config.input_cols && typeof config.input_cols === 'object' && !Array.isArray(config.input_cols)) {
        return config.input_cols;
      }
      return {};
    };
    
    switch (config.transform_type) {
      case 'duplicate':
        return {
          block_type: 'DuplicateColumnsBlock',
          block_config: {
            block_name: config.block_name,
            input_cols: getColumnMapping(),
          },
        };
      
      case 'rename':
        return {
          block_type: 'RenameColumnsBlock',
          block_config: {
            block_name: config.block_name,
            input_cols: getColumnMapping(),
          },
        };
      
      case 'melt':
        return {
          block_type: 'MeltColumnsBlock',
          block_config: {
            block_name: config.block_name,
            input_cols: config.melt_input_cols || config.input_cols || [],
            output_cols: config.melt_output_cols || config.output_cols || [],
          },
        };
      
      default:
        return {
          block_type: 'DuplicateColumnsBlock',
          block_config: {
            block_name: config.block_name,
            input_cols: getColumnMapping(),
          },
        };
    }
  },
};

/**
 * Transform Node Preview Component
 * Displayed in the node on the canvas
 */
export const TransformNodePreview = ({ config }) => {
  const typeLabels = {
    duplicate: 'Duplicate',
    rename: 'Rename',
    melt: 'Melt',
  };

  let details = '';
  if (config.transform_type === 'melt') {
    const cols = config.melt_input_cols || [];
    details = `${cols.length} column(s)`;
  } else {
    const mapping = config.column_mapping || {};
    const count = Object.keys(mapping).length;
    details = `${count} mapping(s)`;
  }

  return (
    <div style={{ padding: '4px 8px', fontSize: '11px' }}>
      <div style={{ color: '#6a6e73', marginBottom: '2px' }}>
        {typeLabels[config.transform_type] || 'Transform'}: {details}
      </div>
      <div style={{ color: '#6753ac', fontWeight: 500 }}>
        Data transformation
      </div>
    </div>
  );
};

export default TransformNodeConfig;
