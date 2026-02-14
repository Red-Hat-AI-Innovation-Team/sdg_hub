import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Alert,
  AlertVariant,
  Badge,
  Grid,
  GridItem,
  Form,
  FormGroup,
  TextInput,
} from '@patternfly/react-core';
import {
  CheckCircleIcon,
} from '@patternfly/react-icons';

/**
 * Step 6: Additional Required Columns
 */
const AdditionalColumnsSection = ({
  expandedSteps,
  toggleStep,
  manualInputColumns,
  additionalColumns,
  needsICL,
  filesWithoutBasicInfoCount,
  setAdditionalColumns,
}) => {
  return (
    <Card style={{ marginBottom: '1rem' }}>
      <CardTitle 
        style={{ cursor: 'pointer' }}
        onClick={() => toggleStep('step6')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {Object.keys(additionalColumns).length === manualInputColumns.length ? (
              <CheckCircleIcon color="var(--pf-global--success-color--100)" />
            ) : (
              <span style={{ 
                width: '24px', 
                height: '24px', 
                borderRadius: '50%', 
                backgroundColor: '#0066cc', 
                color: 'white', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                fontSize: '0.875rem',
                fontWeight: 'bold'
              }}>{needsICL ? '7' : '6'}</span>
            )}
            Additional Required Columns
            <Badge isRead style={{ marginLeft: '0.5rem' }}>{manualInputColumns.length}</Badge>
            {!expandedSteps.step6 && Object.keys(additionalColumns).length === manualInputColumns.length && (
              <Badge style={{ marginLeft: '0.5rem', backgroundColor: '#f0fff0', color: '#3e8635', border: '1px solid #3e8635' }}>
                <CheckCircleIcon style={{ marginRight: '0.25rem' }} />
                Complete
              </Badge>
            )}
          </div>
          <span style={{ fontSize: '0.875rem', color: '#6a6e73' }}>
            {expandedSteps.step6 ? '▼' : '▶'}
          </span>
        </div>
      </CardTitle>
      {expandedSteps.step6 && <CardBody>
        {/* Warning about files without basic info */}
        {filesWithoutBasicInfoCount > 0 && (
          <Alert variant={AlertVariant.warning} isInline title={`${filesWithoutBasicInfoCount} file(s) without basic info`} style={{ marginBottom: '1rem' }}>
            Some chunked files don't have domain and document outline configured in Step 3.
          </Alert>
        )}
        
        <Alert variant={AlertVariant.info} isInline title="Additional Required Columns" style={{ marginBottom: '1rem' }}>
          These columns are required by the selected flow. Enter a value that will be applied to all records.
        </Alert>
        
        <Form>
          <Grid hasGutter>
            {manualInputColumns.map(col => (
              <GridItem key={col} span={6}>
                <FormGroup label={col} fieldId={`col-${col}`} isRequired>
                  <TextInput
                    id={`col-${col}`}
                    value={additionalColumns[col] || ''}
                    onChange={(_, value) => setAdditionalColumns(prev => ({ ...prev, [col]: value }))}
                    placeholder={`Enter value for ${col}...`}
                  />
                </FormGroup>
              </GridItem>
            ))}
          </Grid>
        </Form>
      </CardBody>}
    </Card>
  );
};

export default AdditionalColumnsSection;
