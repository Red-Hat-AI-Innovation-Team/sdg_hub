// SPDX-License-Identifier: Apache-2.0

module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allowed commit types (matches pre-commit config)
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'build',
        'ci',
        'chore',
        'revert',
      ],
    ],
  },
};
