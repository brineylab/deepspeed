# Base Image:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html
# v25.01-py3: Ubuntu 24.04, CUDA 12.8.0, PyTorch 2.6.0 (matches versions on B200s)
# FROM nvcr.io/nvidia/pytorch:25.01-py3
FROM ghcr.io/coreweave/ml-containers/torch:es-cuda-13-dev-4ae9dbb-base-cuda13.1.0-ubuntu22.04-torch2.9.1-vision0.24.1-audio2.9.1-abi1

# NB user
# brineylab UID / GID on B200s
# ARG NB_USER="jovyan"
# ARG NB_UID="1002"
# ARG NB_GID="1002"

# ENV NB_USER=${NB_USER} \
#     NB_UID=${NB_UID} \
#     NB_GID=${NB_GID} \
#     HOME=/home/${NB_USER}

# ENV PATH="$HOME/.local/bin:$PATH"

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

# # Create user and group
# # Grant sudo w/o password
# RUN groupadd -g ${NB_GID} ${NB_USER} && \
#     useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
#     usermod -aG sudo ${NB_USER} && \
#     printf "%s ALL=(ALL) NOPASSWD:ALL\n" "${NB_USER}" > /etc/sudoers.d/${NB_USER} && \
#     chmod 0440 /etc/sudoers.d/${NB_USER}

# # Copy script to fix permissions
# COPY --chown=${NB_UID}:${NB_GID} ./nvidia/deeplearning/fix-permissions /usr/local/bin/fix-permissions
# RUN chmod a+rx /usr/local/bin/fix-permissions && \
#     fix-permissions ${HOME}

# # Working directory & umask
# WORKDIR ${HOME}
# RUN echo 'umask 000' > /etc/profile.d/umask.sh && \
#     echo 'umask 000' >> /etc/bash.bashrc

# Python packages
# USER ${NB_UID}
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
