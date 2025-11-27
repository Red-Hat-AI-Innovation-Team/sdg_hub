import React, { createContext, useContext, useState } from 'react';
import {
  Alert,
  AlertGroup,
  AlertVariant,
  AlertActionCloseButton,
} from '@patternfly/react-core';

/**
 * Notification Context
 * Provides global toast notifications that appear on the right side of the screen
 */
const NotificationContext = createContext();

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  /**
   * Add a notification
   * @param {string} title - Notification title
   * @param {string} variant - 'success', 'danger', 'warning', 'info', 'default'
   * @param {string} description - Optional description
   * @param {number} timeout - Auto-dismiss timeout in ms (0 = no auto-dismiss)
   */
  const addNotification = ({ title, variant = 'info', description = '', timeout = 8000 }) => {
    const id = `notification-${Date.now()}-${Math.random()}`;
    const notification = {
      id,
      title,
      variant,
      description,
    };

    setNotifications(prev => [...prev, notification]);

    // Auto-dismiss after timeout
    if (timeout > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, timeout);
    }

    return id;
  };

  /**
   * Remove a notification
   */
  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  /**
   * Add success notification
   */
  const addSuccessNotification = (title, description, timeout) => {
    return addNotification({ title, variant: 'success', description, timeout });
  };

  /**
   * Add error notification
   */
  const addErrorNotification = (title, description, timeout = 10000) => {
    return addNotification({ title, variant: 'danger', description, timeout });
  };

  /**
   * Add warning notification
   */
  const addWarningNotification = (title, description, timeout) => {
    return addNotification({ title, variant: 'warning', description, timeout });
  };

  /**
   * Add info notification
   */
  const addInfoNotification = (title, description, timeout) => {
    return addNotification({ title, variant: 'info', description, timeout });
  };

  const value = {
    addNotification,
    addSuccessNotification,
    addErrorNotification,
    addWarningNotification,
    addInfoNotification,
    removeNotification,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      
      {/* Toast notifications - fixed position on right side */}
      <AlertGroup 
        isToast 
        isLiveRegion 
        style={{ 
          position: 'fixed', 
          top: '80px', 
          right: '20px', 
          zIndex: 9999,
          maxHeight: 'calc(100vh - 100px)',
          overflowY: 'auto'
        }}
      >
        {notifications.map(notification => (
          <Alert
            key={notification.id}
            variant={notification.variant}
            title={notification.title}
            timeout={true}
            actionClose={
              <AlertActionCloseButton
                title="Close notification"
                onClose={() => removeNotification(notification.id)}
              />
            }
          >
            {notification.description}
          </Alert>
        ))}
      </AlertGroup>
    </NotificationContext.Provider>
  );
};

