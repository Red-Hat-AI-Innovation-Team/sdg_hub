import React from 'react';
import {
  Card,
  CardTitle,
  CardBody,
  Title,
  List,
  ListItem,
  Button,
  Badge,
} from '@patternfly/react-core';
import { PlusCircleIcon } from '@patternfly/react-icons';
import { BLOCK_BUNDLES } from './bundleDefinitions';

/**
 * Bundles Card Component
 * 
 * Shows pre-configured block bundles for common patterns
 */
const BundlesCard = ({ onAddBundle }) => {

  return (
    <Card>
      <CardTitle>
        <Title headingLevel="h3" size="lg">
          📦 Block Bundles
        </Title>
        <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginTop: '0.25rem' }}>
          Pre-configured block sequences for common patterns
        </div>
      </CardTitle>
      <CardBody>
        <List isPlain style={{ display: 'grid', gap: '0.75rem' }}>
          {BLOCK_BUNDLES.map(bundle => (
            <ListItem key={bundle.id}>
              <div style={{
                padding: '1rem',
                background: 'linear-gradient(135deg, #f0f8ff 0%, #e7f1fa 100%)',
                borderRadius: '6px',
                border: '2px solid #0066cc',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: '1rem' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ 
                      fontSize: '1.1rem',
                      fontWeight: 'bold', 
                      marginBottom: '0.5rem',
                      color: '#151515'
                    }}>
                      {bundle.icon} {bundle.name}
                    </div>
                    <div style={{ fontSize: '0.875rem', color: '#6a6e73', marginBottom: '0.75rem' }}>
                      {bundle.description}
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <Badge isRead>{bundle.blockCount} blocks</Badge>
                      <span style={{ fontSize: '0.75rem', color: '#6a6e73' }}>
                        {bundle.generates.join(' → ')}
                      </span>
                    </div>
                  </div>
                  <Button
                    variant="primary"
                    size="sm"
                    icon={<PlusCircleIcon />}
                    onClick={() => onAddBundle({ ...bundle, isBundle: true })}
                  >
                    Add
                  </Button>
                </div>
              </div>
            </ListItem>
          ))}
        </List>
      </CardBody>
    </Card>
  );
};

export default BundlesCard;

