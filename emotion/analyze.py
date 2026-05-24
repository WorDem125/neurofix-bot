import base64
import io
import logging

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align

log = logging.getLogger(__name__)

# словари для перевода меток на русский

EMOTION_RU = {
    "angry":   ("😠", "Злость"),
    "disgust": ("🤢", "Отвращение"),
    "fear":    ("😨", "Страх"),
    "happy":   ("😊", "Радость"),
    "sad":     ("😢", "Грусть"),
    "surprise":("😲", "Удивление"),
    "neutral": ("😐", "Нейтральная"),
}

RACE_RU = {
    "asian":           "Азиатская",
    "black":           "Африканская",
    "indian":          "Южно-азиатская",
    "latino hispanic": "Латиноамериканская",
    "middle eastern":  "Ближневосточная",
    "white":           "Европейская",
}

# инициализация InsightFace (загружается один раз при первом вызове)

_insight_app = None


def _get_insight() -> FaceAnalysis:
    global _insight_app
    if _insight_app is None:
        log.info("InsightFace: загружаю модели...")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _insight_app = app
        log.info("InsightFace: готово")
    return _insight_app


# утилиты форматирования вывода

_BAR_CHARS = " ▏▎▍▌▋▊▉█"

def _bar(pct: float, width: int = 10) -> str:
    total = max(0.0, min(100.0, pct)) * width / 100.0
    full  = int(total)
    frac  = total - full
    result = "█" * full
    idx = round(frac * 8)
    if idx > 0:
        result += _BAR_CHARS[idx]
    return result


def _format_bars(emotion_data: dict) -> str:
    """Возвращает блок с эмоциями для <code>-тега (моноширинный)."""
    MAX_NAME = 11  # "Нейтральная" = 11
    rows = []
    for key, pct in sorted(emotion_data.items(), key=lambda x: x[1], reverse=True):
        if pct < 1:
            continue
        _, name = EMOTION_RU.get(key, ("", key))
        bar  = _bar(pct)
        # выравниваем: имя по левому краю, % по правому (3 символа)
        rows.append(f"{name:<{MAX_NAME}} {pct:>3}%  {bar}")
    return "\n".join(rows)


# основная функция анализа

def analyze_portrait(image_b64: str) -> dict:
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return {"ok": False, "error": "Не удалось декодировать изображение."}

    try:
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return {"ok": False, "error": "Файл повреждён или не является изображением."}

    # детекция лиц + возраст/пол через InsightFace
    try:
        app = _get_insight()
        insight_faces = app.get(img_bgr)
    except Exception as e:
        log.error("InsightFace ошибка: %s", e, exc_info=True)
        return {"ok": False, "error": f"Ошибка анализа: {e}"}

    if not insight_faces:
        return {"ok": False, "error": "Лицо не обнаружено. Попробуйте фото с более чётко видимым лицом."}

    faces = []

    for iface in insight_faces:
        age       = int(round(float(iface.age)))
        gender_ru = "Мужчина" if iface.gender == 1 else "Женщина"

        # эмоции + раса через DeepFace на выровненном лице
        # face_align нормализует по 5 ключевым точкам → точнее чем просто кроп по bbox
        try:
            aligned = face_align.norm_crop(img_bgr, landmark=iface.kps, image_size=224)
        except Exception:
            # kps недоступны — берём простой кроп по bbox
            x1, y1, x2, y2 = iface.bbox.astype(int)
            h, w = img_bgr.shape[:2]
            aligned = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        emotion_data  = {}
        dominant_emotion = "neutral"
        emotion_conf  = 0
        race_ru       = ""
        race_conf     = 0

        try:
            df_results = DeepFace.analyze(
                img_path=aligned,
                actions=["emotion", "race"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            df = df_results[0] if isinstance(df_results, list) else df_results

            em = df.get("emotion", {})
            if isinstance(em, dict):
                dominant_emotion = df.get("dominant_emotion", max(em, key=em.get))
                emotion_conf     = round(em.get(dominant_emotion, 0))
                emotion_data     = {k.lower(): round(v) for k, v in em.items()}

            race_raw = df.get("race", {})
            if isinstance(race_raw, dict):
                dom_race  = df.get("dominant_race", max(race_raw, key=race_raw.get))
                race_conf = round(race_raw.get(dom_race, 0))
                race_ru   = RACE_RU.get(dom_race.lower(), dom_race)

        except Exception as e:
            log.warning("DeepFace ошибка: %s", e)

        emoji, emotion_ru = EMOTION_RU.get(dominant_emotion.lower(), ("🙂", dominant_emotion))

        faces.append({
            "gender":        gender_ru,
            "age":           age,
            "emotion_emoji": emoji,
            "emotion_ru":    emotion_ru,
            "emotion_conf":  emotion_conf,
            "emotion_data":  emotion_data,
            "race":          race_ru,
            "race_conf":     race_conf,
        })

    # форматируем вывод для Telegram
    lines = ["🎭 <b>Анализ портрета</b>"]
    if len(faces) > 1:
        lines.append(f"Обнаружено лиц: {len(faces)}")

    for i, f in enumerate(faces, 1):
        lines.append("")
        if len(faces) > 1:
            lines.append(f"<b>━━ Лицо {i} ━━</b>")

        lines.append(f"{f['emotion_emoji']} <b>Эмоция:</b> {f['emotion_ru']} · {f['emotion_conf']}%")
        lines.append(f"👤 <b>Пол:</b> {f['gender']}")
        lines.append(f"🎂 <b>Возраст:</b> ~{f['age']} лет")
        if f["race"]:
            lines.append(f"🌍 <b>Этнос:</b> {f['race']}")

        if f["emotion_data"]:
            bars = _format_bars(f["emotion_data"])
            lines.append("")
            lines.append("📊 <b>Все эмоции:</b>")
            lines.append(f"<code>{bars}</code>")

    text = "\n".join(lines)
    log.info("Анализ завершён: %d лиц(о)", len(faces))
    return {"ok": True, "text": text, "faces": len(faces)}
