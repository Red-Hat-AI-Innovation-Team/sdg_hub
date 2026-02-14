// SPDX-License-Identifier: Apache-2.0
/**
 * Babel configuration for Jest tests.
 */

module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
    ['@babel/preset-react', { runtime: 'automatic' }],
  ],
};

