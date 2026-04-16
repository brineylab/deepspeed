# Base Image: 
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-12.html
# v25.12-py3: Ubuntu 24.04, CUDA 13.1.0, PyTorch 2.10.0
ARG VERSION=26.01-py3
FROM nvcr.io/nvidia/pytorch:${VERSION}

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        aria2 \
        bash \
        btop \
        curl \
        gh \
        git \
        htop \
        nano \
        sudo \
        tmux \
        tree \
        unzip \
        vim \
        visidata \
        wget \
        zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip install --upgrade pip

# Python packages
RUN pip install --no-cache-dir \
      accelerate \
      deepspeed \
      datasets \
      evaluate \
      matplotlib \
      numpy \
      pandas \
      polars \
      scikit-learn \
      scipy \
      seaborn \
      tokenizers \
      tqdm \
      transformers \
      wandb

# Environment variables
ENV PATH="/usr/local/bin:$PATH" \
    HF_HOME="/workspace/.cache/huggingface" \
    TRITON_CACHE_DIR="/workspace/.triton" \
    WANDB_CACHE_DIR="/workspace/.cache/wandb" \
    WANDB_CONFIG_DIR="/workspace/.cache/wandb-config" \
    WANDB_DATA_DIR="/workspace/.cache/wandb-data" \
    WANDB_ARTIFACT_DIR="/workspace/.cache/wandb-artifact" \
    MPLCONFIGDIR="/workspace/.cache/matplotlib" \
    NUMBA_CACHE_DIR="/workspace/.cache/numba" \
    GIT_CONFIG_GLOBAL="/workspace/.cache/gitconfig"
