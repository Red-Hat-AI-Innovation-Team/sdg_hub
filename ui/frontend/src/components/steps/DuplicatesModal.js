import React from 'react';
import {
  Modal,
  ModalVariant,
  Button,
  Alert,
  AlertVariant,
} from '@patternfly/react-core';

/**
 * Modal for handling duplicate rows detected in uploaded datasets.
 * Gives users the option to remove duplicates or keep all rows.
 */
const DuplicatesModal = ({
  showDuplicatesModal,
  duplicateInfo,
  isRemovingDuplicates,
  onRemoveDuplicates,
  onKeepDuplicates,
}) => {
  return (
    <Modal
      variant={ModalVariant.medium}
      title="Duplicate Rows Detected"
      isOpen={showDuplicatesModal}
      onClose={onKeepDuplicates}
      actions={[
        <Button
          key="remove-duplicates"
          variant="primary"
          onClick={onRemoveDuplicates}
          isLoading={isRemovingDuplicates}
          isDisabled={isRemovingDuplicates}
        >
          Remove Duplicates
        </Button>,
        <Button
          key="keep-duplicates"
          variant="secondary"
          onClick={onKeepDuplicates}
          isDisabled={isRemovingDuplicates}
        >
          Keep All Rows
        </Button>
      ]}
    >
      <Alert
        variant={AlertVariant.warning}
        isInline
        title="Your dataset contains duplicate rows"
        style={{ marginBottom: '1.5rem' }}
      >
        <p style={{ marginTop: '0.5rem' }}>
          We found <strong>{duplicateInfo?.num_duplicates || 0}</strong> duplicate row(s) 
          in your dataset of <strong>{duplicateInfo?.total_rows || 0}</strong> total rows.
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          After removing duplicates, your dataset will have <strong>{duplicateInfo?.unique_rows || 0}</strong> unique rows.
        </p>
      </Alert>

      <div style={{ marginTop: '1.5rem' }}>
        <p style={{ marginBottom: '1rem' }}>
          <strong>Duplicate rows can cause issues during data generation.</strong> We recommend removing them to ensure better quality results.
        </p>
        
        <div style={{ 
          padding: '1rem',
          backgroundColor: '#e7f5e1',
          borderRadius: '4px',
          marginBottom: '1rem',
          borderLeft: '4px solid #3e8635'
        }}>
          <p style={{ marginBottom: '0.5rem' }}>
            <strong>Remove Duplicates (Recommended)</strong>
          </p>
          <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            Remove duplicate rows, keeping only the first occurrence of each unique row.
            This will reduce your dataset from {duplicateInfo?.total_rows || 0} to {duplicateInfo?.unique_rows || 0} rows.
          </p>
        </div>
        
        <div style={{ 
          padding: '1rem',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px'
        }}>
          <p style={{ marginBottom: '0.5rem' }}>
            <strong>Keep All Rows</strong>
          </p>
          <p style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            Continue with all rows including duplicates. Note that this may result in 
            redundant data generation and potentially lower quality output.
          </p>
        </div>
      </div>
    </Modal>
  );
};

export default DuplicatesModal;
