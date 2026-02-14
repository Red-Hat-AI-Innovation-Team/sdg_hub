import React, { createContext, useContext } from 'react';

const ExecutionContext = createContext(null);

export const ExecutionProvider = ({ children, value }) => (
  <ExecutionContext.Provider value={value}>
    {children}
  </ExecutionContext.Provider>
);

export const useExecutionConfig = () => {
  const context = useContext(ExecutionContext);
  if (!context) {
    // Return safe defaults if used outside provider (backward compat)
    return { selectedFlow: null, modelConfig: {}, datasetConfig: {} };
  }
  return context;
};

export default ExecutionContext;
