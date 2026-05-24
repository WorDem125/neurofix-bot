FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    "tensorflow[and-cuda]>=2.18.0" \
    "deepface>=0.0.89" \
    "tf-keras" \
    "retina-face>=0.0.17" \
    "insightface>=0.7.3" \
    "onnxruntime-gpu>=1.16.0" \
    "opencv-python-headless>=4.8.0" \
    "redis>=5.0.0" \
    "python-dotenv>=1.0.0"

# прописываем CUDA-библиотеки установленные через pip в динамический линковщик
RUN find /usr/local/lib/python3.10/dist-packages/nvidia -mindepth 2 -maxdepth 2 \
        -type d -name "lib" \
    | tee /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

COPY . .

CMD ["python3", "main.py"]
