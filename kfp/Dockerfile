# SPDX-License-Identifier: Apache-2.0
# SDG Hub KFP Component Container Image
#
# Build: docker build -t sdg-hub-kfp:dev -f kfp/Dockerfile .
# For production, use UBI 9 base image

FROM python:3.11-slim

LABEL maintainer="Red Hat AI Innovation Team"
LABEL description="SDG Hub KFP Component for synthetic data generation"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . /app

# Install sdg_hub with kfp dependencies
RUN pip install --no-cache-dir ".[kfp]"

# Verify installation
RUN python -c "from sdg_hub.core import Flow; print('SDG Hub installed successfully')"

# No ENTRYPOINT - KFP will set the command