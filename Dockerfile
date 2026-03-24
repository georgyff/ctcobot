FROM python:3.12-slim

# Switch to root to install system packages
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (latest LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /mnt

# Expose JupyterLab port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
ENV UV_SYSTEM_PYTHON=1