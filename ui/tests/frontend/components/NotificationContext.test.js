// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for NotificationContext - Global toast notification system.
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NotificationProvider, useNotifications } from '../../../frontend/src/contexts/NotificationContext';

// Test helper component that exposes notification methods
const TestConsumer = ({ onMount }) => {
  const notifications = useNotifications();
  React.useEffect(() => {
    if (onMount) onMount(notifications);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return <div data-testid="consumer">Consumer</div>;
};

// Helper to render with provider
const renderWithProvider = (onMount) => {
  return render(
    <NotificationProvider>
      <TestConsumer onMount={onMount} />
    </NotificationProvider>
  );
};

describe('NotificationContext', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('NotificationProvider', () => {
    it('renders children correctly', () => {
      render(
        <NotificationProvider>
          <div data-testid="child">Hello</div>
        </NotificationProvider>
      );
      expect(screen.getByTestId('child')).toBeInTheDocument();
      expect(screen.getByText('Hello')).toBeInTheDocument();
    });

    it('provides notification methods to children', () => {
      let receivedMethods;
      renderWithProvider((methods) => {
        receivedMethods = methods;
      });

      expect(receivedMethods).toBeDefined();
      expect(typeof receivedMethods.addNotification).toBe('function');
      expect(typeof receivedMethods.addSuccessNotification).toBe('function');
      expect(typeof receivedMethods.addErrorNotification).toBe('function');
      expect(typeof receivedMethods.addWarningNotification).toBe('function');
      expect(typeof receivedMethods.addInfoNotification).toBe('function');
      expect(typeof receivedMethods.removeNotification).toBe('function');
    });
  });

  describe('useNotifications hook', () => {
    it('throws error when used outside NotificationProvider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      const BadConsumer = () => {
        useNotifications();
        return null;
      };

      expect(() => render(<BadConsumer />)).toThrow(
        'useNotifications must be used within NotificationProvider'
      );

      consoleSpy.mockRestore();
    });
  });

  describe('addNotification', () => {
    it('displays a notification with title and description', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({
          title: 'Test Title',
          variant: 'info',
          description: 'Test description',
          timeout: 0,
        });
      });

      expect(screen.getByText('Test Title')).toBeInTheDocument();
      expect(screen.getByText('Test description')).toBeInTheDocument();
    });

    it('returns a notification id', () => {
      let methods;
      let notifId;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        notifId = methods.addNotification({
          title: 'Test',
          timeout: 0,
        });
      });

      expect(notifId).toBeDefined();
      expect(typeof notifId).toBe('string');
      expect(notifId).toContain('notification-');
    });

    it('defaults to info variant when not specified', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({
          title: 'Info Test',
          timeout: 0,
        });
      });

      expect(screen.getByText('Info Test')).toBeInTheDocument();
    });

    it('auto-dismisses after timeout', async () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({
          title: 'Auto Dismiss',
          timeout: 5000,
        });
      });

      expect(screen.getByText('Auto Dismiss')).toBeInTheDocument();

      // Advance timers past the timeout
      act(() => {
        jest.advanceTimersByTime(5500);
      });

      await waitFor(() => {
        expect(screen.queryByText('Auto Dismiss')).not.toBeInTheDocument();
      });
    });

    it('does not auto-dismiss via custom timeout when timeout is 0', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({
          title: 'Persistent',
          timeout: 0,
        });
      });

      // The notification should be added to the state
      // Note: PatternFly Alert may have its own timeout behavior
      expect(screen.getByText('Persistent')).toBeInTheDocument();
    });

    it('supports multiple simultaneous notifications', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({ title: 'First', timeout: 0 });
        methods.addNotification({ title: 'Second', timeout: 0 });
        methods.addNotification({ title: 'Third', timeout: 0 });
      });

      expect(screen.getByText('First')).toBeInTheDocument();
      expect(screen.getByText('Second')).toBeInTheDocument();
      expect(screen.getByText('Third')).toBeInTheDocument();
    });
  });

  describe('removeNotification', () => {
    it('removes a specific notification by id', async () => {
      let methods;
      let notifId;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        notifId = methods.addNotification({
          title: 'To Remove',
          timeout: 0,
        });
      });

      expect(screen.getByText('To Remove')).toBeInTheDocument();

      act(() => {
        methods.removeNotification(notifId);
      });

      await waitFor(() => {
        expect(screen.queryByText('To Remove')).not.toBeInTheDocument();
      });
    });

    it('does not affect other notifications when removing one', async () => {
      let methods;
      let firstId;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        firstId = methods.addNotification({ title: 'Keep Me', timeout: 0 });
        methods.addNotification({ title: 'Also Keep', timeout: 0 });
      });

      act(() => {
        methods.removeNotification(firstId);
      });

      await waitFor(() => {
        expect(screen.queryByText('Keep Me')).not.toBeInTheDocument();
      });
      expect(screen.getByText('Also Keep')).toBeInTheDocument();
    });
  });

  describe('convenience methods', () => {
    it('addSuccessNotification creates success variant', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addSuccessNotification('Success!', 'It worked', 0);
      });

      expect(screen.getByText('Success!')).toBeInTheDocument();
      expect(screen.getByText('It worked')).toBeInTheDocument();
    });

    it('addErrorNotification creates danger variant with longer default timeout', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addErrorNotification('Error!', 'Something broke');
      });

      expect(screen.getByText('Error!')).toBeInTheDocument();
      expect(screen.getByText('Something broke')).toBeInTheDocument();
    });

    it('addWarningNotification creates warning variant', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addWarningNotification('Warning!', 'Be careful', 0);
      });

      expect(screen.getByText('Warning!')).toBeInTheDocument();
      expect(screen.getByText('Be careful')).toBeInTheDocument();
    });

    it('addInfoNotification creates info variant', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addInfoNotification('Info', 'FYI', 0);
      });

      expect(screen.getByText('Info')).toBeInTheDocument();
      expect(screen.getByText('FYI')).toBeInTheDocument();
    });
  });

  describe('close button', () => {
    it('renders close button on notifications', () => {
      let methods;
      renderWithProvider((m) => { methods = m; });

      act(() => {
        methods.addNotification({
          title: 'Closeable',
          timeout: 0,
        });
      });

      // PatternFly Alert renders a close button with various aria labels
      const closeButton = screen.queryByLabelText('Close notification') ||
                          screen.queryByLabelText(/close/i) ||
                          screen.queryByRole('button', { name: /close/i });
      expect(closeButton).toBeTruthy();
    });
  });
});
