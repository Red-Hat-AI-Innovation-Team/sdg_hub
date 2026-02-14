import React, { useState } from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Tabs,
  Tab,
  TabTitleText,
  Alert,
  AlertVariant,
  Label,
  Title,
  Divider,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  InProgressIcon,
  ArrowRightIcon,
  AngleDownIcon,
  AngleRightIcon,
} from '@patternfly/react-icons';

/**
 * Format a value for display as readable text
 */
const formatValue = (value) => {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value, null, 2);
};

/**
 * A single row value with expand/collapse for long content
 */
const RowValue = ({ value, rowIndex, totalRows }) => {
  const [expanded, setExpanded] = useState(false);
  const formatted = formatValue(value);
  const isLong = formatted.length > 150;
  const preview = isLong ? formatted.slice(0, 150) + '...' : formatted;

  return (
    <div style={{
      padding: '10px 14px',
      background: '#fff',
      borderBottom: '1px solid #f0f0f0',
    }}>
      {totalRows > 1 && (
        <div style={{
          fontSize: '11px',
          fontWeight: 700,
          color: '#6a6e73',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          marginBottom: '6px',
        }}>
          Row {rowIndex + 1}
        </div>
      )}
      {isLong ? (
        <>
          <div style={{
            fontSize: '13px',
            lineHeight: '1.65',
            color: '#151515',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontFamily: "'Red Hat Mono', 'SFMono-Regular', Consolas, monospace",
            maxHeight: expanded ? '200px' : 'none',
            overflowY: expanded ? 'auto' : 'hidden',
            borderRadius: expanded ? '4px' : '0',
            background: expanded ? '#fafafa' : 'transparent',
            padding: expanded ? '8px' : '0',
          }}>
            {expanded ? formatted : preview}
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              marginTop: '6px',
              padding: '3px 10px',
              fontSize: '12px',
              color: '#0066cc',
              background: '#e7f1fa',
              border: '1px solid #bee1f4',
              borderRadius: '10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            {expanded ? 'Collapse' : `Show all (${formatted.length} chars)`}
          </button>
        </>
      ) : (
        <div style={{
          fontSize: '13px',
          lineHeight: '1.65',
          color: '#151515',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontFamily: "'Red Hat Mono', 'SFMono-Regular', Consolas, monospace",
        }}>
          {formatted}
        </div>
      )}
    </div>
  );
};

/**
 * A column section card showing column name + all row values
 */
const ColumnCard = ({ columnName, data, isNew }) => {
  const values = (data || []).map(row => row?.[columnName]);

  return (
    <div style={{
      borderRadius: '8px',
      border: '1px solid #d2d2d2',
      overflow: 'hidden',
      borderLeft: isNew ? '4px solid #3e8635' : '4px solid #0066cc',
    }}>
      {/* Column header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '10px 14px',
        background: isNew ? '#f3faf2' : '#f0f4f9',
        borderBottom: '1px solid #d2d2d2',
        gap: '8px',
      }}>
        <span style={{
          fontWeight: 700,
          fontSize: '14px',
          fontFamily: "'Red Hat Mono', 'SFMono-Regular', Consolas, monospace",
          color: '#151515',
        }}>
          {columnName}
        </span>
        {isNew && <Label color="green" isCompact>new</Label>}
        <span style={{
          marginLeft: 'auto',
          fontSize: '12px',
          color: '#6a6e73',
          fontWeight: 500,
        }}>
          {values.length} row{values.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Row values */}
      {values.length === 0 ? (
        <div style={{ padding: '14px', color: '#6a6e73', fontStyle: 'italic' }}>No data</div>
      ) : (
        values.map((val, i) => (
          <RowValue key={i} value={val} rowIndex={i} totalRows={values.length} />
        ))
      )}
    </div>
  );
};

/**
 * Collapsible section for intermediate/other columns (prompts, metadata).
 * Collapsed by default so the user sees the primary output first.
 */
const OtherColumnsSection = ({ columns, data }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div style={{
      borderRadius: '8px',
      border: '1px solid #d2d2d2',
      overflow: 'hidden',
      background: '#fafafa',
    }}>
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          padding: '10px 14px',
          background: '#f5f5f5',
          border: 'none',
          cursor: 'pointer',
          gap: '8px',
          fontFamily: 'inherit',
          fontSize: '13px',
          color: '#6a6e73',
        }}
      >
        {expanded ? <AngleDownIcon /> : <AngleRightIcon />}
        <span>Other columns added ({columns.length})</span>
        <span style={{ marginLeft: 'auto', fontSize: '12px' }}>
          {columns.join(', ')}
        </span>
      </button>
      {expanded && (
        <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {columns.map(col => (
            <ColumnCard
              key={col}
              columnName={col}
              data={data}
              isNew={true}
            />
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Split new_columns into primary output vs intermediate columns based on node type.
 *
 * We rely on node TYPE and block expansion structure, NOT on column names
 * (since names are user-defined in the custom flow creator).
 *
 * - LLM nodes always expand to 3 sub-blocks: PromptBuilder → LLMChat → Extractor.
 *   The columns are added in that order, so the LAST new column is always the
 *   extracted model output. The others are intermediate (prompt, raw response).
 *
 * - Parser / Eval / Transform: ALL new columns are the actual output.
 */
const splitNewColumns = (nodeType, newColumns) => {
  if (!newColumns || newColumns.length === 0) {
    return { primary: [], other: [] };
  }

  if (nodeType === 'llm' && newColumns.length > 1) {
    // LLM expands to 3 sub-blocks: PromptBuilder → LLMChat → Extractor
    // Last column = extracted content (primary), rest = intermediate
    return {
      primary: [newColumns[newColumns.length - 1]],
      other: newColumns.slice(0, -1),
    };
  }

  if (nodeType === 'eval' && newColumns.length > 2) {
    // Eval expands to 5 sub-blocks: PromptBuilder → LLMChat → Extractor → Parser → Filter
    // Last 2 columns = judgment + explanation (primary), rest = intermediate
    return {
      primary: newColumns.slice(-2),
      other: newColumns.slice(0, -2),
    };
  }

  // Parser, transform, or anything else: all columns are primary output
  return { primary: newColumns, other: [] };
};

/**
 * Extract the relevant input column names from a node's config.
 * Only shows the columns this node actually works with.
 */
const getNodeInputColumnNames = (nodeData) => {
  const config = nodeData?.config || {};
  let cols = [];

  // LLM nodes: input_cols is an array like ['document', 'document_outline']
  if (Array.isArray(config.input_cols)) {
    cols = config.input_cols;
  }
  // Parser/some nodes: input_cols is a single string
  else if (typeof config.input_cols === 'string' && config.input_cols) {
    cols = [config.input_cols];
  }
  // Transform nodes: input_cols might be an object (column mapping)
  else if (typeof config.input_cols === 'object' && config.input_cols) {
    cols = Object.keys(config.input_cols);
  }

  return cols.filter(Boolean);
};

/**
 * Node I/O Modal
 * 
 * Displays the input and output data for a node during/after test execution.
 * Shows only the relevant input columns and new columns with readable formatting.
 */
const NodeIOModal = ({
  isOpen,
  onClose,
  nodeData,
  testResult,
}) => {
  const [activeTab, setActiveTab] = useState(0);

  if (!nodeData || !testResult) {
    return null;
  }

  const { status, input_data, output_data, input_columns, output_columns, new_columns, error, execution_time_ms } = testResult;

  const getStatusIcon = () => {
    switch (status) {
      case 'success':
      case 'complete':
        return <CheckCircleIcon color="#3e8635" />;
      case 'error':
        return <ExclamationCircleIcon color="#c9190b" />;
      case 'running':
        return <InProgressIcon color="#0066cc" className="pf-v5-u-spin" />;
      case 'skipped':
        return <ExclamationCircleIcon color="#f0ab00" />;
      default:
        return null;
    }
  };

  // Determine which input columns to display.
  // First try: columns from the node config (what the user configured).
  // But the visual config names (e.g. "ex1") may differ from actual data column names
  // (e.g. "extract_Python_content") because the serializer rewires connections.
  // So we verify they actually exist in the data, and fall back to all input columns
  // minus new_columns if they don't.
  const configInputCols = getNodeInputColumnNames(nodeData);
  const actualDataCols = input_data && input_data.length > 0 ? Object.keys(input_data[0]) : (input_columns || []);
  const matchedConfigCols = configInputCols.filter(col => actualDataCols.includes(col));

  const displayInputCols = matchedConfigCols.length > 0
    ? matchedConfigCols
    : (input_columns || []).filter(col => !(new_columns || []).includes(col));

  const newCols = new_columns || [];

  // Split new columns into primary output vs. intermediate based on node type
  const { primary: primaryNewCols, other: otherNewCols } = splitNewColumns(nodeData?.type, newCols);

  return (
    <Modal
      variant={ModalVariant.large}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {getStatusIcon()}
          <span>{nodeData.label || nodeData.config?.block_name || 'Node'}</span>
          <Label color="blue" isCompact>{nodeData.type}</Label>
        </div>
      }
      isOpen={isOpen}
      onClose={onClose}
      style={{ '--pf-v5-c-modal-box--MaxHeight': '90vh' }}
      actions={[
        <Button key="close" variant="primary" onClick={onClose}>
          Close
        </Button>,
      ]}
    >
      <div style={{ overflowY: 'auto', maxHeight: 'calc(90vh - 180px)', paddingRight: '4px' }}>
      {/* Skipped Alert */}
      {status === 'skipped' && (
        <Alert
          variant={AlertVariant.info}
          isInline
          title="Block was skipped"
          style={{ marginBottom: '16px' }}
        >
          {testResult.skipped_reason || 'A previous block removed all samples, so this block was not executed.'}
        </Alert>
      )}

      {/* Error Alert */}
      {status === 'error' && error && (
        <Alert
          variant={AlertVariant.danger}
          isInline
          title="Block execution failed"
          style={{ marginBottom: '16px' }}
        >
          {error}
        </Alert>
      )}

      {/* Execution Summary */}
      <DescriptionList isHorizontal isCompact style={{ marginBottom: '16px' }}>
        <DescriptionListGroup>
          <DescriptionListTerm>Status</DescriptionListTerm>
          <DescriptionListDescription>
            <Label color={status === 'success' || status === 'complete' ? 'green' : status === 'error' ? 'red' : status === 'skipped' ? 'orange' : 'blue'}>
              {status}
            </Label>
          </DescriptionListDescription>
        </DescriptionListGroup>
        {execution_time_ms !== undefined && (
          <DescriptionListGroup>
            <DescriptionListTerm>Execution Time</DescriptionListTerm>
            <DescriptionListDescription>{execution_time_ms}ms</DescriptionListDescription>
          </DescriptionListGroup>
        )}
        <DescriptionListGroup>
          <DescriptionListTerm>Rows</DescriptionListTerm>
          <DescriptionListDescription>
            {testResult.input_rows || input_data?.length || 0}
            <ArrowRightIcon style={{ margin: '0 8px', opacity: 0.5 }} />
            {testResult.output_rows || output_data?.length || 0}
          </DescriptionListDescription>
        </DescriptionListGroup>
      </DescriptionList>

      <Divider style={{ marginBottom: '16px' }} />

      {/* Tabs */}
      <Tabs
        activeKey={activeTab}
        onSelect={(event, tabIndex) => setActiveTab(tabIndex)}
        aria-label="Node I/O tabs"
      >
        {/* Input Columns Tab - only relevant columns */}
        <Tab eventKey={0} title={<TabTitleText>Input Columns ({displayInputCols.length})</TabTitleText>}>
          <div style={{
            marginTop: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
          }}>
            {displayInputCols.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: '#6a6e73', fontStyle: 'italic' }}>
                No input columns for this node
              </div>
            ) : (
              displayInputCols.map(col => (
                <ColumnCard
                  key={col}
                  columnName={col}
                  data={input_data}
                  isNew={false}
                />
              ))
            )}
          </div>
        </Tab>

        {/* Output Tab -- always show if block completed (even with 0 rows) */}
        {status !== 'skipped' && (
          <Tab eventKey={1} title={
            <TabTitleText>
              Output
              {primaryNewCols.length > 0 && (
                <Label color="green" isCompact style={{ marginLeft: '8px' }}>
                  {primaryNewCols.length}
                </Label>
              )}
              {primaryNewCols.length === 0 && (
                <Label color="orange" isCompact style={{ marginLeft: '8px' }}>
                  0 rows
                </Label>
              )}
            </TabTitleText>
          }>
            <div style={{
              marginTop: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}>
              {/* Primary output columns */}
              {primaryNewCols.map(col => (
                <ColumnCard
                  key={col}
                  columnName={col}
                  data={output_data}
                  isNew={true}
                />
              ))}

              {/* No output: explain why */}
              {primaryNewCols.length === 0 && (!output_data || output_data.length === 0) && (
                <Alert
                  variant={AlertVariant.warning}
                  isInline
                  title="No output rows produced"
                  style={{ marginBottom: '8px' }}
                >
                  {nodeData?.type === 'parser'
                    ? 'The parser removed all rows. This usually means the input text did not contain the configured start/end tags. Check the Input tab to see what was received and verify your tag configuration matches the model output format.'
                    : nodeData?.type === 'eval'
                    ? 'The evaluation filter removed all rows. The sample did not meet the passing criteria.'
                    : 'This block produced 0 output rows.'}
                </Alert>
              )}

              {/* Intermediate columns (prompt, raw response) -- collapsed */}
              {otherNewCols.length > 0 && (
                <OtherColumnsSection
                  columns={otherNewCols}
                  data={output_data}
                />
              )}
            </div>
          </Tab>
        )}
      </Tabs>
      </div>
    </Modal>
  );
};

export default NodeIOModal;
