FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    git

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    numpy \
    scipy \
    matplotlib \
    sklearn \
    tensorflow

# Expose ports for TensorBoard
EXPOSE 6006

WORKDIR = "/ml_algorithms"

