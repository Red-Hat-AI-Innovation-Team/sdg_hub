import React from 'react';
import { Button } from '@patternfly/react-core';
import { BarsIcon } from '@patternfly/react-icons';

/**
 * Custom App Header Component
 * Dark header bar with Red Hat branding and navigation toggle
 */
const AppHeader = ({ isSidebarOpen, onToggleSidebar }) => {
  return (
    <div style={{
      backgroundColor: '#151515',
      color: 'white',
      padding: '12px 16px',
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      borderBottom: '1px solid #d2d2d2',
      position: 'sticky',
      top: 0,
      zIndex: 1000,
      width: '100%'
    }}>
      {/* Hamburger Menu */}
      <Button
        variant="plain"
        onClick={onToggleSidebar}
        style={{ color: 'white', padding: '8px' }}
        aria-label="Toggle navigation"
      >
        <BarsIcon style={{ fontSize: '20px' }} />
      </Button>

      {/* Branding */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        <span style={{ 
          fontSize: '1.5rem',
          fontWeight: 700,
          lineHeight: 1.2
        }}>
          SDG Hub
        </span>
        <span style={{ 
          fontSize: '0.875rem',
          fontWeight: 400,
          lineHeight: 1.3,
          opacity: 0.9
        }}>
          Synthetic Data Generation
        </span>
      </div>
    </div>
  );
};

export default AppHeader;

