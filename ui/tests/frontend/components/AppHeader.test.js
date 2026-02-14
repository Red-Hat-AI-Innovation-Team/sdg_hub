// SPDX-License-Identifier: Apache-2.0
/**
 * Tests for AppHeader component - Application header bar.
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AppHeader from '../../../frontend/src/components/AppHeader';

describe('AppHeader', () => {
  const defaultProps = {
    isSidebarOpen: true,
    onToggleSidebar: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the application title', () => {
    render(<AppHeader {...defaultProps} />);
    expect(screen.getByText('SDG Hub')).toBeInTheDocument();
  });

  it('renders the subtitle', () => {
    render(<AppHeader {...defaultProps} />);
    expect(screen.getByText('Synthetic Data Generation')).toBeInTheDocument();
  });

  it('renders the toggle navigation button', () => {
    render(<AppHeader {...defaultProps} />);
    const toggleButton = screen.getByLabelText('Toggle navigation');
    expect(toggleButton).toBeInTheDocument();
  });

  it('calls onToggleSidebar when toggle button is clicked', async () => {
    const user = userEvent.setup();
    render(<AppHeader {...defaultProps} />);

    await user.click(screen.getByLabelText('Toggle navigation'));
    expect(defaultProps.onToggleSidebar).toHaveBeenCalledTimes(1);
  });

  it('renders with dark background styling', () => {
    const { container } = render(<AppHeader {...defaultProps} />);
    const header = container.firstChild;
    expect(header).toHaveStyle({ backgroundColor: '#151515' });
  });
});
