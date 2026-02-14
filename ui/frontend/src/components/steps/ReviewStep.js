import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  Alert,
  AlertVariant,
  Grid,
  GridItem,
} from '@patternfly/react-core';
import { CheckCircleIcon } from '@patternfly/react-icons';

/**
 * Review Step Component
 * 
 * Allows users to:
 * - Review all configuration settings before saving
 */
const ReviewStep = ({ 
  selectedFlow, 
  modelConfig, 
  datasetConfig, 
  onError,
}) => {

  if (!selectedFlow || !modelConfig || !datasetConfig) {
    return (
      <Alert
        variant={AlertVariant.warning}
        isInline
        title="Incomplete configuration"
      >
        Please complete all previous steps before reviewing the configuration.
      </Alert>
    );
  }

  return (
    <Grid hasGutter style={{ height: '100%' }}>
      {/* Configuration Summary */}
      <GridItem span={12} style={{ display: 'flex', flexDirection: 'column' }}>
        <Card style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CardTitle>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Title headingLevel="h2" size="xl">
                Configuration Summary
              </Title>
            </div>
          </CardTitle>
          <CardBody style={{ flex: 1, overflowY: 'auto' }}>
            <Grid hasGutter>
              {/* Flow Information */}
              <GridItem span={6}>
                <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                  <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" style={{ marginRight: '0.5rem' }} />
                  Flow
                </Title>
                <DescriptionList isHorizontal isCompact>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Name</DescriptionListTerm>
                    <DescriptionListDescription>
                      {selectedFlow.name}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>ID</DescriptionListTerm>
                    <DescriptionListDescription>
                      <code>{selectedFlow.id}</code>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Version</DescriptionListTerm>
                    <DescriptionListDescription>
                      {selectedFlow.version}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </GridItem>

              {/* Model Configuration */}
              <GridItem span={6}>
                <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                  <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" style={{ marginRight: '0.5rem' }} />
                  Model
                </Title>
                <DescriptionList isHorizontal isCompact>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Model</DescriptionListTerm>
                    <DescriptionListDescription>
                      <code>{modelConfig.model}</code>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>API Base</DescriptionListTerm>
                    <DescriptionListDescription>
                      <code>{modelConfig.api_base}</code>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>API Key</DescriptionListTerm>
                    <DescriptionListDescription>
                      {'*'.repeat(modelConfig.api_key?.length || 0)}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  {modelConfig.additional_params && Object.keys(modelConfig.additional_params).length > 0 && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Additional Params</DescriptionListTerm>
                      <DescriptionListDescription>
                        {Object.entries(modelConfig.additional_params).map(([key, value]) => (
                          <div key={key}>
                            <code>{key}: {value}</code>
                          </div>
                        ))}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}
                </DescriptionList>
              </GridItem>

              {/* Dataset Configuration */}
              <GridItem span={6}>
                <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                  <CheckCircleIcon color="var(--pf-v5-global--success-color--100)" style={{ marginRight: '0.5rem' }} />
                  Dataset
                </Title>
                <DescriptionList isHorizontal isCompact>
                  {datasetConfig.uploaded_file && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Source</DescriptionListTerm>
                      <DescriptionListDescription>
                        Uploaded: <code>{datasetConfig.uploaded_file}</code>
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}
                  <DescriptionListGroup>
                    <DescriptionListTerm>Data Files</DescriptionListTerm>
                    <DescriptionListDescription>
                      <code>{datasetConfig.data_files}</code>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Split</DescriptionListTerm>
                    <DescriptionListDescription>
                      {datasetConfig.split}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  {datasetConfig.num_samples && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Num Samples</DescriptionListTerm>
                      <DescriptionListDescription>
                        {datasetConfig.num_samples}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}
                  <DescriptionListGroup>
                    <DescriptionListTerm>Shuffle</DescriptionListTerm>
                    <DescriptionListDescription>
                      {datasetConfig.shuffle ? 'Yes' : 'No'}
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  {datasetConfig.shuffle && (
                    <DescriptionListGroup>
                      <DescriptionListTerm>Seed</DescriptionListTerm>
                      <DescriptionListDescription>
                        {datasetConfig.seed}
                      </DescriptionListDescription>
                    </DescriptionListGroup>
                  )}
                </DescriptionList>
              </GridItem>
            </Grid>
          </CardBody>
        </Card>
      </GridItem>

      {/* Save Options Explanation */}
      <GridItem span={12}>
        <Alert
          variant={AlertVariant.info}
          isInline
          title="Choose how to proceed"
        >
          <p style={{ marginBottom: '0.75rem' }}>
            Your configuration is ready. Choose one of the following options:
          </p>
          <ul style={{ marginLeft: '1.5rem', marginBottom: '0' }}>
            <li style={{ marginBottom: '0.5rem' }}>
              <strong>Run Now</strong> — Saves your configuration and immediately starts data generation. 
              You can monitor progress in the Flow Runs History.
            </li>
            <li>
              <strong>Save for Later</strong> — Saves your configuration without running. 
              You can start it anytime from the Data Generation Flows page.
            </li>
          </ul>
        </Alert>
      </GridItem>
    </Grid>
  );
};

export default ReviewStep;
