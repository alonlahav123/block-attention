FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gosu \
    openssh-server \
    python3 \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-venv \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

RUN mkdir -p /var/run/sshd /workspace /root/.ssh

COPY docker/entrypoint.sh /usr/local/bin/block-attention-entrypoint.sh
RUN chmod +x /usr/local/bin/block-attention-entrypoint.sh

WORKDIR /workspace

EXPOSE 22 8080

ENTRYPOINT ["/usr/local/bin/block-attention-entrypoint.sh"]
CMD ["bash", "-lc", "sleep infinity"]
