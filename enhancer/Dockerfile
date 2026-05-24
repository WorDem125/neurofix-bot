FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

# Real-ESRGAN: клонируем и устанавливаем
RUN git clone https://github.com/xinntao/Real-ESRGAN /app/engines/realesrgan
RUN pip install --no-cache-dir basicsr facexlib gfpgan
RUN pip install --no-cache-dir -e /app/engines/realesrgan

# Фикс: basicsr 1.4.2 использует torchvision.transforms.functional_tensor — удалён в новых версиях
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /opt/conda/lib/python3.10/site-packages/basicsr/data/degradations.py

# CodeFormer: клонируем
RUN git clone https://github.com/sczhou/CodeFormer /app/engines/codeformer

# Фикс: создаём version.py ДО установки зависимостей CodeFormer (они обновляют numpy до 2.x)
RUN printf '__version__ = "1.4.2"\n__gitsha__ = "unknown"\n' \
    > /app/engines/codeformer/basicsr/version.py

RUN pip install --no-cache-dir -r /app/engines/codeformer/requirements.txt

# Фикс повторно — зависимости CodeFormer могут переустановить basicsr
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /opt/conda/lib/python3.10/site-packages/basicsr/data/degradations.py 2>/dev/null || true

# Скачиваем модели CodeFormer в /app/weights/ — именно оттуда их ищет CodeFormer
RUN mkdir -p /app/weights/facelib /app/weights/realesrgan /app/weights/CodeFormer && \
    wget -q -O /app/weights/facelib/parsing_parsenet.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth && \
    wget -q -O /app/weights/facelib/detection_Resnet50_Final.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /app/weights/facelib/detection_mobilenet0.25_Final.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth && \
    wget -q -O /app/weights/realesrgan/RealESRGAN_x2plus.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ВАЖНО: numpy<2 фиксируем в конце — зависимости CodeFormer обновляют его до 2.x, что ломает PyTorch 2.2.2
RUN pip install --no-cache-dir 'numpy>=1.24,<2.0'

COPY . .

CMD ["python", "main.py"]
