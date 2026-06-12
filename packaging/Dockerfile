ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libceres-dev \
    libeigen3-dev \
    libopencv-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip
RUN pip install \
    "matplotlib>=3.6" \
    "numpy>=1.23" \
    opencv-python \
    "pyyaml>=6.0"
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . /app

RUN cmake -S stereo_calib -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j"$(nproc)"
RUN python3 -m py_compile \
    run_calibrate_once.py \
    run_pipeline.py \
    stereo_calib_py/config.py \
    stereo_calib_py/pipeline.py

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libceres-dev \
    libopencv-dev \
    python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/run_calibrate_once.py /app/run_calibrate_once.py
COPY --from=builder /app/stereo_calib_py /app/stereo_calib_py
COPY --from=builder /app/stereo_calib/scripts /app/stereo_calib/scripts
COPY --from=builder /app/build/bin /app/build/bin
COPY --from=builder /app/templates /app/templates

ENV PATH="/opt/venv/bin:${PATH}"
ENV MPLBACKEND=Agg

ENTRYPOINT ["python3", "run_calibrate_once.py"]
