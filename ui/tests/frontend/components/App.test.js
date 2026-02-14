// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for App component - Main application shell.
 * Tests navigation, health checks, sidebar, page routing.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NotificationProvider } from '../../../frontend/src/contexts/NotificationContext';

// Mock the services/api module
jest.mock('../../../frontend/src/services/api', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
  },
  healthCheck: jest.fn(),
  executionAPI: {
    getGenerationStatus: jest.fn().mockResolvedValue({ running_generations: [] }),
  },
  runsAPI: {
    list: jest.fn().mockResolvedValue({ data: [] }),
    update: jest.fn().mockResolvedValue({}),
  },
  API_BASE_URL: 'http://localhost:8000',
  flowAPI: {
    listFlows: jest.fn().mockResolvedValue({ data: { flows: [] } }),
    listFlowsWithDetails: jest.fn().mockResolvedValue({ data: { flows: {} } }),
    searchFlows: jest.fn().mockResolvedValue({ data: { flows: [] } }),
  },
  savedConfigAPI: {
    list: jest.fn().mockResolvedValue({ data: [] }),
  },
}));

// Mock child components to keep tests focused on App shell behavior
jest.mock('../../../frontend/src/components/HomeDashboard', () => {
  return function MockHomeDashboard({ onNavigate }) {
    return (
      <div data-testid="home-dashboard">
        <button onClick={() => onNavigate('flows')}>Go to Flows</button>
        <button onClick={() => onNavigate('configure-flow', { flow: 'test' })}>
          Configure Flow
        </button>
      </div>
    );
  };
});

jest.mock('../../../frontend/src/components/Dashboard', () => {
  return function MockDashboard() {
    return <div data-testid="dashboard-page">Dashboard</div>;
  };
});

jest.mock('../../../frontend/src/components/DataGenerationFlowsPage', () => {
  return function MockFlowsPage({ onNavigate, onEditConfiguration }) {
    return (
      <div data-testid="flows-page">
        <button onClick={() => onNavigate('configure-flow')}>New Flow</button>
        <button onClick={() => onEditConfiguration({ id: '123', flow_name: 'Test' })}>
          Edit Config
        </button>
      </div>
    );
  };
});

jest.mock('../../../frontend/src/components/FlowRunsHistoryPage', () => {
  return function MockRunsPage() {
    return <div data-testid="runs-page">Flow Runs History</div>;
  };
});

jest.mock('../../../frontend/src/components/UnifiedFlowWizard', () => {
  return function MockWizard({ onComplete, onCancel }) {
    return (
      <div data-testid="wizard-page">
        <button onClick={() => onComplete({})}>Complete</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    );
  };
});

jest.mock('../../../frontend/src/components/AppHeader', () => {
  return function MockAppHeader({ isSidebarOpen, onToggleSidebar }) {
    return (
      <div data-testid="app-header">
        <button data-testid="sidebar-toggle" onClick={onToggleSidebar}>
          Toggle Sidebar
        </button>
        <span data-testid="sidebar-state">{isSidebarOpen ? 'open' : 'closed'}</span>
      </div>
    );
  };
});

import App from '../../../frontend/src/App';
import { healthCheck, executionAPI } from '../../../frontend/src/services/api';

// Helper to render App with NotificationProvider
const renderApp = () => {
  return render(
    <NotificationProvider>
      <App />
    </NotificationProvider>
  );
};

describe('App Component', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    sessionStorage.clear();
    localStorage.clear();
    // Default: health check succeeds
    healthCheck.mockResolvedValue({ status: 'healthy' });
    executionAPI.getGenerationStatus.mockResolvedValue({ running_generations: [] });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('health check', () => {
    it('shows loading spinner while checking backend health', () => {
      // Make health check hang
      healthCheck.mockReturnValue(new Promise(() => {}));
      renderApp();

      expect(screen.getByText('Connecting to SDG Hub API...')).toBeInTheDocument();
    });

    it('renders main app when backend is healthy', async () => {
      healthCheck.mockResolvedValue({ status: 'healthy' });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });
    });

    it('shows error when backend health check fails', async () => {
      healthCheck.mockRejectedValue(new Error('Connection refused'));
      renderApp();

      await waitFor(() => {
        expect(screen.getByText(/Cannot connect to backend API/i)).toBeInTheDocument();
      });
    });

    it('shows reconnecting message for network errors', async () => {
      healthCheck.mockRejectedValue(new Error('Network Error'));
      renderApp();

      await waitFor(() => {
        expect(screen.getByText(/Reconnecting to server/i)).toBeInTheDocument();
      });
    });
  });

  describe('navigation', () => {
    it('shows Home page by default', async () => {
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('home-dashboard')).toBeInTheDocument();
      });
    });

    it('navigates to Dashboard when clicked', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Dashboard'));

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });

    it('navigates to Flow Runs History when clicked', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Flow Runs History'));

      await waitFor(() => {
        expect(screen.getByTestId('runs-page')).toBeInTheDocument();
      });
    });

    it('navigates from Home to Flows page via child component', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('home-dashboard')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Go to Flows'));

      await waitFor(() => {
        expect(screen.getByTestId('flows-page')).toBeInTheDocument();
      });
    });

    it('persists active page to sessionStorage', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Flow Runs History'));

      await waitFor(() => {
        expect(sessionStorage.getItem('app_active_item')).toBe('flow-runs');
      });
    });
  });

  describe('sidebar', () => {
    it('sidebar is open by default', async () => {
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('sidebar-state')).toHaveTextContent('open');
      });
    });

    it('toggles sidebar when button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('sidebar-state')).toHaveTextContent('open');
      });

      await user.click(screen.getByTestId('sidebar-toggle'));

      await waitFor(() => {
        expect(screen.getByTestId('sidebar-state')).toHaveTextContent('closed');
      });
    });

    it('renders navigation items', async () => {
      renderApp();

      await waitFor(() => {
        expect(screen.getByText('Home')).toBeInTheDocument();
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Data Generation Flows')).toBeInTheDocument();
        expect(screen.getByText('Flow Runs History')).toBeInTheDocument();
      });
    });
  });

  describe('wizard flow', () => {
    it('opens wizard when navigating to configure-flow', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('home-dashboard')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Configure Flow'));

      await waitFor(() => {
        expect(screen.getByTestId('wizard-page')).toBeInTheDocument();
      });
    });

    it('returns to flows page when wizard completes', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('home-dashboard')).toBeInTheDocument();
      });

      // Navigate to wizard
      await user.click(screen.getByText('Configure Flow'));
      await waitFor(() => {
        expect(screen.getByTestId('wizard-page')).toBeInTheDocument();
      });

      // Complete wizard
      await user.click(screen.getByText('Complete'));

      await waitFor(() => {
        expect(screen.getByTestId('flows-page')).toBeInTheDocument();
      });
    });

    it('returns to flows page when wizard is cancelled', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('home-dashboard')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Configure Flow'));
      await waitFor(() => {
        expect(screen.getByTestId('wizard-page')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Cancel'));

      await waitFor(() => {
        expect(screen.getByTestId('flows-page')).toBeInTheDocument();
      });
    });
  });

  describe('execution state management', () => {
    it('loads execution states from localStorage on mount', async () => {
      localStorage.setItem('sdg_execution_states', JSON.stringify({
        'config-1': {
          configId: 'config-1',
          status: 'completed',
          isRunning: false,
        }
      }));

      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });
      // App should not crash and should render
    });

    it('handles corrupted localStorage gracefully', async () => {
      localStorage.setItem('sdg_execution_states', 'not-valid-json');
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      renderApp();

      await waitFor(() => {
        expect(screen.getByTestId('app-header')).toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });
  });
});
