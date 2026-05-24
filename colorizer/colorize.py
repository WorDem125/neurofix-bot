import base64
import io
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/DDColor")

log = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from ddcolor.pipeline import ColorizationPipeline, build_ddcolor_model
        from ddcolor.model import DDColor
        from huggingface_hub import hf_hub_download

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("DDColor: загружаю на %s", device)

        model_path = hf_hub_download(
            repo_id="piddnad/ddcolor_artistic",
            filename="pytorch_model.bin",
        )

        model = build_ddcolor_model(
            DDColor,
            model_path=model_path,
            input_size=512,
            model_size="large",
            device=device,
        )

        _pipeline = ColorizationPipeline(model=model, input_size=512, device=device)
        log.info("DDColor artistic: модель загружена")
    return _pipeline


def colorize_image(image_b64: str) -> dict:
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return {"ok": False, "error": "Не удалось декодировать изображение."}

    try:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return {"ok": False, "error": "Файл повреждён или не является изображением."}

    w, h = img_pil.size
    log.info("colorize: %dx%d", w, h)

    # убираем сепию и тонировку — DDColor ожидает чистый grayscale
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    try:
        pipeline = _get_pipeline()
        colorized_bgr = pipeline.process(img_bgr_gray)
    except Exception as e:
        log.error("DDColor ошибка: %s", e, exc_info=True)
        return {"ok": False, "error": f"Ошибка раскраски: {e}"}

    # усиливаем насыщенность если DDColor дал слишком бледный результат
    b_out, g_out, r_out = cv2.split(colorized_bgr)
    rg = abs(float(r_out.mean()) - float(g_out.mean()))
    rb = abs(float(r_out.mean()) - float(b_out.mean()))
    gb = abs(float(g_out.mean()) - float(b_out.mean()))
    max_diff = max(rg, rb, gb)
    log.info("OUTPUT B=%.1f G=%.1f R=%.1f  max_diff=%.1f",
             b_out.mean(), g_out.mean(), r_out.mean(), max_diff)

    if max_diff >= 20:
        log.info("vivid (%.1f), boost пропущен", max_diff)
    else:
        boost = 1.3 if max_diff >= 10 else 1.8
        log.info("boost x%.1f (max_diff=%.1f)", boost, max_diff)
        lab = cv2.cvtColor(colorized_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        lab[:, :, 1] = np.clip(lab[:, :, 1] * boost, -127, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * boost, -127, 127)
        colorized_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr * 255, 0, 255).astype(np.uint8)

    # ограничиваем размер — Telegram не принимает фото больше 4096px
    max_side = max(colorized_bgr.shape[:2])
    if max_side > 4096:
        scale = 4096 / max_side
        new_w = int(colorized_bgr.shape[1] * scale)
        new_h = int(colorized_bgr.shape[0] * scale)
        colorized_bgr = cv2.resize(colorized_bgr, (new_w, new_h),
                                   interpolation=cv2.INTER_LANCZOS4)

    ok, buf = cv2.imencode(".jpg", colorized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"ok": False, "error": "Не удалось сохранить результат."}

    result_b64 = base64.b64encode(buf.tobytes()).decode()
    log.info("colorize OK: %dx%d → %dx%d",
             w, h, colorized_bgr.shape[1], colorized_bgr.shape[0])
    return {"ok": True, "image_b64": result_b64}
