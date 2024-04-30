FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for DUSt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git=1:2.34.1-1ubuntu1.10 \
    libglib2.0-0=2.72.4-0ubuntu2.2 \
    libgl1-mesa-glx \
    libglu1-mesa \
    python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp_install

COPY dust3r/requirements.txt dust3r/requirements_optional.txt /tmp_install/
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74

COPY dust3r/croco/ /tmp_install/croco/

WORKDIR /tmp_install/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
