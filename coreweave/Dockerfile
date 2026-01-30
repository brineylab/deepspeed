FROM ghcr.io/coreweave/ml-containers/torch:es-cuda-13-dev-4ae9dbb-base-cuda13.1.0-ubuntu22.04-torch2.9.1-vision0.24.1-audio2.9.1-abi1

# System packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        bash \
        btop \
        curl \
        htop \
        nano \
        sudo \
        tmux \
        unzip \
        vim \
        wget \
        zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python packages
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        accelerate \
        "deepspeed<0.17.6" \
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
