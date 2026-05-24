FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/piddnad/DDColor.git /DDColor

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    "numpy<2.0" \
    "basicsr" \
    "einops" \
    "timm" \
    "transformers>=4.20.0" \
    "huggingface_hub>=0.16.0" \
    "opencv-python-headless>=4.5.0,<4.9" \
    "redis>=5.0.0" \
    "python-dotenv>=1.0.0"

COPY . .

CMD ["python", "main.py"]
