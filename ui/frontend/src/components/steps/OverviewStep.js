import React, { useState } from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  Grid,
  GridItem,
  List,
  ListItem,
  Alert,
  AlertVariant,
  FileUpload,
  Button,
  Divider,
} from '@patternfly/react-core';
import { UploadIcon, CheckCircleIcon, InfoCircleIcon } from '@patternfly/react-icons';
import { configAPI, flowAPI } from '../../services/api';

/**
 * Overview Step Component
 * 
 * Provides an introduction to SDG Hub and the configuration UI.
 * Allows users to import a previously exported configuration for quick setup.
 */
const OverviewStep = ({ onConfigImport, onFlowSelect, onError }) => {
  // Import config state
  const [importFile, setImportFile] = useState(null);
  const [importFileName, setImportFileName] = useState('');
  const [isImporting, setIsImporting] = useState(false);

  /**
   * Handle configuration file import
   */
  const handleImportConfig = async () => {
    try {
      setIsImporting(true);
      
      // Create File object from uploaded content
      const blob = new Blob([importFile], { type: 'application/json' });
      const file = new File([blob], importFileName, { type: 'application/json' });
      
      // Import configuration
      const response = await configAPI.importConfig(file);
      
      // Select the imported flow
      const flowInfo = await flowAPI.getFlowInfo(response.flow.name);
      onFlowSelect(flowInfo);
      
      // Notify parent about imported configuration
      if (onConfigImport) {
        onConfigImport(response.imported_config);
      }
      
      // Clear import file
      setImportFile(null);
      setImportFileName('');
      
    } catch (error) {
      onError('Failed to import configuration: ' + error.message);
    } finally {
      setIsImporting(false);
    }
  };

  /**
   * Handle import file change
   */
  const handleImportFileChange = async (event, file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setImportFile(e.target.result);
      setImportFileName(file.name);
    };
    reader.onerror = () => {
      onError('Failed to read configuration file');
    };
    reader.readAsText(file);
  };

  /**
   * Clear import file
   */
  const handleClearImport = () => {
    setImportFile(null);
    setImportFileName('');
  };

  return (
    <Grid hasGutter>
      {/* Welcome Section */}
      <GridItem span={12}>
        <Card>
          <CardTitle>
            <Title headingLevel="h1" size="2xl">
              Welcome to SDG Hub Configuration UI
            </Title>
          </CardTitle>
          <CardBody>
            <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem' }}>
              Configure synthetic data generation pipelines without writing code. 
              This wizard guides you through selecting flows, configuring models, loading datasets, 
              and testing your configuration.
            </p>
          </CardBody>
        </Card>
      </GridItem>

      {/* About SDG Hub */}
      <GridItem span={6}>
        <Card isFullHeight>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              <InfoCircleIcon style={{ marginRight: '0.5rem', color: '#0066cc' }} />
              About SDG Hub
            </Title>
          </CardTitle>
          <CardBody>
            <p style={{ marginBottom: '1rem' }}>
              <strong>SDG Hub</strong> is a modular Python framework for building synthetic data 
              generation pipelines using composable blocks and flows.
            </p>
            
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
              Key Features:
            </Title>
            <List>
              <ListItem>
                <strong>Modular Composability</strong> - Mix and match blocks like Lego pieces
              </ListItem>
              <ListItem>
                <strong>Async Performance</strong> - High-throughput LLM processing
              </ListItem>
              <ListItem>
                <strong>Built-in Validation</strong> - Type safety and error handling
              </ListItem>
              <ListItem>
                <strong>Auto-Discovery</strong> - Automatic flow and block registration
              </ListItem>
              <ListItem>
                <strong>Rich Monitoring</strong> - Detailed logging and progress tracking
              </ListItem>
            </List>
          </CardBody>
        </Card>
      </GridItem>

      {/* How to Use This UI */}
      <GridItem span={6}>
        <Card isFullHeight>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              <CheckCircleIcon style={{ marginRight: '0.5rem', color: '#3e8635' }} />
              How to Use This UI
            </Title>
          </CardTitle>
          <CardBody>
            <Title headingLevel="h4" size="md" style={{ marginBottom: '0.5rem' }}>
              Configuration Steps:
            </Title>
            <List isPlain>
              <ListItem>
                <strong>1. Select Flow</strong> - Choose from 7 available data generation flows
              </ListItem>
              <ListItem>
                <strong>2. Configure Model</strong> - Set up your LLM endpoint and parameters
              </ListItem>
              <ListItem>
                <strong>3. Configure Dataset</strong> - Upload or specify your seed data
              </ListItem>
              <ListItem>
                <strong>4. Review & Confirm</strong> - Verify all settings before testing
              </ListItem>
              <ListItem>
                <strong>5. Dry Run (Optional)</strong> - Test with a small sample and view real-time logs
              </ListItem>
            </List>

            <Alert
              variant={AlertVariant.info}
              isInline
              title="Pro Tip"
              style={{ marginTop: '1.5rem' }}
            >
              Export your configuration after setup to quickly reuse it later!
            </Alert>
          </CardBody>
        </Card>
      </GridItem>

      {/* Quick Start Options */}
      <GridItem span={12}>
        <Divider />
      </GridItem>

      <GridItem span={12}>
        <Card>
          <CardTitle>
            <Title headingLevel="h2" size="xl">
              Get Started
            </Title>
          </CardTitle>
          <CardBody>
            <Grid hasGutter>
              <GridItem span={6}>
                <Card style={{ height: '100%', border: '2px solid #0066cc' }}>
                  <CardBody style={{ padding: '2rem' }}>
                    <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                      🚀 New Configuration
                    </Title>
                    <p style={{ marginBottom: '1rem', minHeight: '3rem' }}>
                      Start fresh and configure a new flow from scratch. 
                      We'll guide you through each step.
                    </p>
                    <p style={{ fontSize: '0.875rem', color: '#6a6e73', marginTop: 'auto' }}>
                      <strong>Click "Next"</strong> to begin selecting a flow →
                    </p>
                  </CardBody>
                </Card>
              </GridItem>

              <GridItem span={6}>
                <Card style={{ height: '100%', border: '2px solid #3e8635' }}>
                  <CardBody style={{ padding: '2rem' }}>
                    <Title headingLevel="h3" size="lg" style={{ marginBottom: '1rem' }}>
                      📥 Import Existing Configuration
                    </Title>
                    <p style={{ marginBottom: '1rem', minHeight: '3rem' }}>
                      Have a previously exported configuration? Upload it below to 
                      auto-fill all settings and jump directly to the review step.
                    </p>
                    
                    <FileUpload
                      id="config-import"
                      type="text"
                      value={importFile}
                      filename={importFileName}
                      filenamePlaceholder="Drop config JSON here or click to browse"
                      onFileInputChange={handleImportFileChange}
                      onClearClick={handleClearImport}
                      browseButtonText="Browse"
                      clearButtonText="Clear"
                      isLoading={isImporting}
                      allowEditingUploadedText={false}
                      dropzoneProps={{
                        accept: { 'application/json': ['.json'] }
                      }}
                    />
                    <Button
                      variant="primary"
                      onClick={handleImportConfig}
                      isDisabled={!importFile}
                      isLoading={isImporting}
                      icon={<UploadIcon />}
                      style={{ marginTop: '1rem', width: '100%' }}
                    >
                      Import & Auto-Configure
                    </Button>
                  </CardBody>
                </Card>
              </GridItem>
            </Grid>
          </CardBody>
        </Card>
      </GridItem>

    </Grid>
  );
};

export default OverviewStep;

