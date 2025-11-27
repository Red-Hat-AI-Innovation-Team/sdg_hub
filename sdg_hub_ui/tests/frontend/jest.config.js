// SPDX-License-Identifier: Apache-2.0
/**
 * Jest configuration for SDG Hub UI Frontend tests.
 */

module.exports = {
  // Test environment
  testEnvironment: 'jsdom',
  
  // Setup files
  setupFilesAfterEnv: ['<rootDir>/setupTests.js'],
  
  // Test patterns
  testMatch: [
    '**/__tests__/**/*.[jt]s?(x)',
    '**/?(*.)+(spec|test).[jt]s?(x)',
  ],
  
  // Module name mapping for imports
  moduleNameMapper: {
    // Handle CSS imports
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Handle image imports
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/__mocks__/fileMock.js',
    // Handle module aliases
    '^@/(.*)$': '<rootDir>/../../src/$1',
    // Force use of parent's React to avoid multiple instances
    '^react$': '<rootDir>/../../node_modules/react',
    '^react-dom$': '<rootDir>/../../node_modules/react-dom',
    '^react-dom/client$': '<rootDir>/../../node_modules/react-dom/client',
  },
  
  // Transform files
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
  
  // Ignore patterns
  transformIgnorePatterns: [
    '/node_modules/(?!(@patternfly|axios)/)',
  ],
  
  // Coverage configuration
  collectCoverageFrom: [
    '../../frontend/src/**/*.{js,jsx}',
    '!../../frontend/src/index.js',
    '!../../frontend/src/reportWebVitals.js',
    '!../../frontend/src/**/*.d.ts',
  ],
  
  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50,
    },
  },
  
  // Root directories
  roots: ['<rootDir>'],
  
  // Module directories
  moduleDirectories: ['node_modules', '../../frontend/node_modules'],
  
  // Test timeout
  testTimeout: 10000,
  
  // Verbose output
  verbose: true,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks automatically
  restoreMocks: true,
};

