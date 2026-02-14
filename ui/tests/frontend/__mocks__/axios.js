// SPDX-License-Identifier: Apache-2.0
/**
 * Mock axios module for testing API calls.
 */

const mockAxios = {
  create: jest.fn(() => mockAxios),
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
  interceptors: {
    request: {
      use: jest.fn(),
      eject: jest.fn(),
    },
    response: {
      use: jest.fn(),
      eject: jest.fn(),
    },
  },
  defaults: {
    headers: {
      common: {},
    },
  },
};

// Helper to reset all mocks
mockAxios.reset = () => {
  mockAxios.get.mockReset();
  mockAxios.post.mockReset();
  mockAxios.put.mockReset();
  mockAxios.delete.mockReset();
};

// Helper to mock successful responses
mockAxios.mockSuccess = (data, method = 'get') => {
  mockAxios[method].mockResolvedValueOnce({ data, status: 200 });
};

// Helper to mock error responses
mockAxios.mockError = (message, status = 500, method = 'get') => {
  mockAxios[method].mockRejectedValueOnce({
    response: {
      data: { detail: message },
      status,
    },
    message,
  });
};

export default mockAxios;

