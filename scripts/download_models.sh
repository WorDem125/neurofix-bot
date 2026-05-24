#!/bin/bash
# Загрузка моделей перед первым запуском
set -e

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

echo "Загрузка моделей в $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"/{realesrgan,codeformer/codeformer}

# shape predictor для dlib (определение ключевых точек лица)
if [ ! -f "$MODELS_DIR/shape_predictor_68_face_landmarks.dat" ]; then
    echo "↓ shape_predictor_68_face_landmarks.dat..."
    wget -q --show-progress \
        -O /tmp/shape_predictor.dat.bz2 \
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d /tmp/shape_predictor.dat.bz2
    mv /tmp/shape_predictor.dat "$MODELS_DIR/shape_predictor_68_face_landmarks.dat"
    echo "✓ shape_predictor готов"
fi

# Real-ESRGAN: веса для апскейла ×4
if [ ! -f "$MODELS_DIR/realesrgan/RealESRGAN_x4plus.pth" ]; then
    echo "↓ RealESRGAN_x4plus.pth..."
    wget -q --show-progress \
        -O "$MODELS_DIR/realesrgan/RealESRGAN_x4plus.pth" \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    echo "✓ RealESRGAN готов"
fi

# CodeFormer: веса для восстановления лиц
if [ ! -f "$MODELS_DIR/codeformer/codeformer.pth" ]; then
    echo "↓ codeformer.pth..."
    wget -q --show-progress \
        -O "$MODELS_DIR/codeformer/codeformer.pth" \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    cp "$MODELS_DIR/codeformer/codeformer.pth" \
       "$MODELS_DIR/codeformer/codeformer/codeformer.pth"
    echo "✓ CodeFormer готов"
fi

# модели реставрации (Global stage — восстановление текстуры)
WORKER_GLOBAL="$(cd "$(dirname "$0")/.." && pwd)/worker/engines/old_photo/Global/checkpoints"
if [ ! -d "$WORKER_GLOBAL" ] || [ -z "$(ls -A "$WORKER_GLOBAL" 2>/dev/null)" ]; then
    echo "↓ Модели реставрации (Global)..."
    mkdir -p "$(dirname "$WORKER_GLOBAL")"
    wget -q --show-progress \
        -O /tmp/global_checkpoints.zip \
        https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
    unzip -q /tmp/global_checkpoints.zip -d "$(dirname "$WORKER_GLOBAL")"
    rm /tmp/global_checkpoints.zip
    echo "✓ Global checkpoints готовы"
fi

# модели реставрации (Face Enhancement stage — восстановление лиц)
WORKER_FACE="$(cd "$(dirname "$0")/.." && pwd)/worker/engines/old_photo/Face_Enhancement/checkpoints"
if [ ! -d "$WORKER_FACE" ] || [ -z "$(ls -A "$WORKER_FACE" 2>/dev/null)" ]; then
    echo "↓ Модели реставрации (Face Enhancement)..."
    mkdir -p "$(dirname "$WORKER_FACE")"
    wget -q --show-progress \
        -O /tmp/face_checkpoints.zip \
        https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
    unzip -q /tmp/face_checkpoints.zip -d "$(dirname "$WORKER_FACE")"
    rm /tmp/face_checkpoints.zip
    echo "✓ Face Enhancement checkpoints готовы"
fi

echo ""
echo "✓ Все модели загружены"
echo ""
echo "Следующие модели загрузятся автоматически при первом запуске:"
echo "  • DDColor        — HuggingFace: piddnad/ddcolor_artistic"
echo "  • OpenCLIP       — HuggingFace: laion/CLIP-ViT-B-32"
echo "  • InsightFace    — InsightFace CDN"
echo "  • DeepFace       — GitHub releases"
