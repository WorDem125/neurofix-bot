import base64
import io
import logging

import cv2
import open_clip
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _get_model():
    global _model, _preprocess, _tokenizer, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("OpenCLIP: загружаю модель на %s", _device)
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _model = _model.to(_device).eval()
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        log.info("OpenCLIP: модель загружена")
    return _model, _preprocess, _tokenizer, _device


def _clip_probs(image_pil: Image.Image, labels: list[str]) -> list[float]:
    model, preprocess, tokenizer, device = _get_model()
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    text_tokens = tokenizer(labels).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        image_features = model.encode_image(img_tensor)
        text_features  = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.T).squeeze(0)
        probs  = logits.softmax(dim=-1).cpu().numpy()
    return probs.tolist()


def analyze_image(image_b64: str) -> dict:
    # декодируем base64 и проверяем что это вообще изображение
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return _invalid("corrupt", "Не удалось декодировать файл — отправьте другое изображение.")

    try:
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_pil.verify()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        return _invalid("corrupt", "Файл повреждён или не является изображением.")

    w, h = img_pil.size

    if w < 80 or h < 80:
        return _invalid("too_small", f"Изображение слишком маленькое ({w}×{h} px).")

    arr = np.array(img_pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # быстрые проверки без нейросети: яркость, энтропия, документ
    mean_brightness = float(gray.mean())
    if mean_brightness < 12:
        return _invalid("too_dark", "Изображение полностью чёрное — загрузите нормальное фото.")
    if mean_brightness > 240:
        return _invalid("blank", "Изображение пустое или почти белое.")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-9)))
    if entropy < 3.5:
        return _invalid("low_content", "Изображение однородное — нечего обрабатывать.")

    white_ratio = float(np.mean(gray > 210))
    edges = cv2.Canny(gray, 80, 200)
    edge_density = float(edges.mean())

    if white_ratio > 0.45 and edge_density < 6.0 and mean_brightness > 200:
        return _invalid("blank", "Изображение почти пустое — загрузите нормальное фото.")

    if white_ratio > 0.50 and edge_density > 5.0 and mean_brightness > 180:
        return _invalid("document", "Это похоже на документ или скан текста.\nЗагрузите обычную фотографию.")

    # определяем цветность через CLIP
    bw_labels = [
        "a black and white photograph",
        "a sepia toned vintage photograph",
        "a color photograph",
    ]
    bw_probs = _clip_probs(img_pil, bw_labels)
    is_bw = (bw_probs[0] + bw_probs[1]) > bw_probs[2]
    log.info("bw probs: bw=%.3f sepia=%.3f color=%.3f → is_bw=%s",
             bw_probs[0], bw_probs[1], bw_probs[2], is_bw)

    # детекция повреждений через CLIP
    damage_labels = [
        "a photograph with visible white scratches, torn edges, missing pieces or water stains",
        "a photograph without scratches, tears or physical damage",
    ]
    dmg_probs = _clip_probs(img_pil, damage_labels)
    has_damage = dmg_probs[0] > 0.50
    log.info("damage probs: damaged=%.3f clean=%.3f → has_damage=%s",
             dmg_probs[0], dmg_probs[1], has_damage)

    # детекция лица — запускаем последней чтобы не ломать логику для не-портретов
    face_labels = [
        "a portrait photo with a human face",
        "a photo without any human faces",
    ]
    face_probs = _clip_probs(img_pil, face_labels)
    has_faces = face_probs[0] > face_probs[1]
    log.info("face probs: portrait=%.3f other=%.3f → has_faces=%s",
             face_probs[0], face_probs[1], has_faces)

    route = _determine_route(is_bw, has_faces, has_damage)

    log.info("analyze OK: %dx%d bw=%s faces=%s damage=%s route=%s",
             w, h, is_bw, has_faces, has_damage, route)

    return {
        "valid":      True,
        "is_bw":      is_bw,
        "has_faces":  has_faces,
        "has_damage": has_damage,
        "width":      w,
        "height":     h,
        "route":      route,
    }


def _invalid(error_type: str, message: str) -> dict:
    return {"valid": False, "error_type": error_type, "description": message}


def _determine_route(is_bw: bool, has_faces: bool, has_damage: bool) -> str:
    if has_damage:
        return "restore"
    if is_bw:
        return "colorize"
    return "enhance"
