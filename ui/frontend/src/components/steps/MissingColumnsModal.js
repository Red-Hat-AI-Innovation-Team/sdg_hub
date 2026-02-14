import React from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Alert,
  AlertVariant,
  List,
  ListItem,
  Form,
  FormGroup,
  TextInput,
  Title,
} from '@patternfly/react-core';

/**
 * Modal for handling missing required columns in uploaded datasets.
 * Contains two modals:
 * 1. Initial modal showing which columns are missing with options
 * 2. Step-by-step modal for filling in each missing column value
 */
const MissingColumnsModal = ({
  missingColumns,
  showMissingColumnsModal,
  addingMissingColumns,
  currentMissingColumnIndex,
  missingColumnValues,
  currentColumnInput,
  numSamples,
  onUseRepetitiveFormat,
  onCancelAndFixManually,
  onSaveColumnValue,
  onCancelAdding,
  onColumnInputChange,
}) => {
  return (
    <>
      {/* Missing Columns Modal */}
      <Modal
        variant={ModalVariant.medium}
        title="Missing Required Columns"
        isOpen={showMissingColumnsModal}
        onClose={onCancelAndFixManually}
        actions={[
          <Button
            key="use-repetitive"
            variant="primary"
            onClick={onUseRepetitiveFormat}
          >
            Use Repetitive Format
          </Button>,
          <Button
            key="cancel"
            variant="secondary"
            onClick={onCancelAndFixManually}
          >
            Cancel
          </Button>
        ]}
      >
        <Alert
          variant={AlertVariant.warning}
          isInline
          title="Your dataset does not contain all of the required columns"
          style={{ marginBottom: '1.5rem' }}
        >
          <p style={{ marginTop: '0.5rem' }}>
            The following columns are missing from your dataset:
          </p>
          <List isPlain style={{ marginTop: '0.75rem', marginLeft: '1rem' }}>
            {missingColumns.map(col => (
              <ListItem key={col}>
                <code style={{ 
                  backgroundColor: '#fff3cd',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  color: '#856404'
                }}>
                  {col}
                </code>
              </ListItem>
            ))}
          </List>
        </Alert>

        <div style={{ marginTop: '1.5rem' }}>
          <p style={{ marginBottom: '1rem' }}>
            <strong>You have two options:</strong>
          </p>
          
          <div style={{ 
            padding: '1rem',
            backgroundColor: '#f5f5f5',
            borderRadius: '4px',
            marginBottom: '1rem'
          }}>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>1. Use Repetitive Format</strong>
            </p>
            <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
              Add the same content to all rows in your dataset for each missing column. 
              You'll be guided step by step to fill in each missing column.
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem',
            backgroundColor: '#f5f5f5',
            borderRadius: '4px'
          }}>
            <p style={{ marginBottom: '0.5rem' }}>
              <strong>2. Cancel and Work Manually</strong>
            </p>
            <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
              Fix your dataset file manually by adding the missing columns, 
              then upload it again.
            </p>
          </div>
        </div>
      </Modal>

      {/* Add Missing Columns Step-by-Step Modal */}
      <Modal
        variant={ModalVariant.medium}
        title={`Add Missing Column: ${missingColumns[currentMissingColumnIndex]}`}
        isOpen={addingMissingColumns}
        onClose={onCancelAdding}
        actions={[
          <Button
            key="save"
            variant="primary"
            onClick={() => {
              const columnName = missingColumns[currentMissingColumnIndex];
              onSaveColumnValue(columnName, currentColumnInput);
            }}
          >
            {currentMissingColumnIndex < missingColumns.length - 1 ? 'Next Column' : 'Finish & Load Dataset'}
          </Button>,
          <Button
            key="cancel"
            variant="secondary"
            onClick={onCancelAdding}
          >
            Cancel
          </Button>
        ]}
      >
        <Alert
          variant={AlertVariant.info}
          isInline
          title={`Column ${currentMissingColumnIndex + 1} of ${missingColumns.length}`}
          style={{ marginBottom: '1.5rem' }}
        >
          <p>
            This value will be added to <strong>all {numSamples} samples</strong> in your dataset.
          </p>
        </Alert>

        <Form>
          <FormGroup
            label={`Value for "${missingColumns[currentMissingColumnIndex]}"`}
            isRequired
            fieldId="missing-column-value"
            helperText="Enter the value that will be used for all rows in this column"
          >
            <TextInput
              isRequired
              type="text"
              id="missing-column-value"
              name="missing-column-value"
              value={currentColumnInput}
              onChange={(event, value) => onColumnInputChange(value)}
              placeholder={`Enter value for ${missingColumns[currentMissingColumnIndex]}`}
            />
          </FormGroup>
        </Form>

        {Object.keys(missingColumnValues).length > 0 && (
          <div style={{ marginTop: '1.5rem' }}>
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
              Previously Added Columns
            </Title>
            <List isPlain isBordered>
              {Object.entries(missingColumnValues).map(([col, val]) => (
                <ListItem key={col}>
                  <code>{col}</code>: <strong>{val}</strong>
                </ListItem>
              ))}
            </List>
          </div>
        )}
      </Modal>
    </>
  );
};

export default MissingColumnsModal;
