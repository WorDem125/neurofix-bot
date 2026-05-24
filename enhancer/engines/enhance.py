import sys
import uuid
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

REALESRGAN_SCRIPT = Path("/app/engines/realesrgan/inference_realesrgan.py")
CODEFORMER_SCRIPT = Path("/app/engines/codeformer/inference_codeformer.py")
REALESRGAN_MODEL  = Path("/models/realesrgan/RealESRGAN_x4plus.pth")
CODEFORMER_MODEL  = Path("/models/codeformer/codeformer.pth")
OUTPUT_BASE = Path("/data/results")

# CodeFormer ищет веса относительно /app/ — кладём симлинки в нужные пути
APP_WEIGHTS = Path("/app/weights")

MAX_TELEGRAM_PX = 4096
JPEG_QUALITY    = 90


def _ensure_codeformer_weights() -> None:
    """Создаёт симлинки из volume-mounted моделей в пути, которые ожидает CodeFormer."""
    cf_dir = APP_WEIGHTS / "CodeFormer"
    cf_dir.mkdir(parents=True, exist_ok=True)
    link = cf_dir / "codeformer.pth"
    if not link.exists():
        link.symlink_to(CODEFORMER_MODEL)
    (APP_WEIGHTS / "facelib").mkdir(parents=True, exist_ok=True)
    (APP_WEIGHTS / "realesrgan").mkdir(parents=True, exist_ok=True)


def _run_enhance(input_path: Path, output_dir: Path, fidelity: float, upscale: int) -> Path | None:
    """Полный пайплайн CodeFormer: восстановление лиц + апскейл фона через Real-ESRGAN."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_codeformer_weights()

    cmd = [
        sys.executable, str(CODEFORMER_SCRIPT),
        "-w", str(fidelity),
        "-s", str(upscale),
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--bg_upsampler", "realesrgan",
        "--face_upsample",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd="/app")
    if r.stdout:
        log.info("Enhance: %s", r.stdout[-1000:])
    if r.returncode != 0:
        log.error("Enhance завершился с ошибкой: %s", r.stderr[-500:])
        return None

    final_dir = output_dir / "final_results"
    results = sorted(final_dir.glob("*.*")) if final_dir.exists() else []
    if not results:
        log.error("Enhance: файл результата не найден в final_results/")
        return None

    log.info("Enhance готово: %s", results[0].name)
    return results[0]


def _run_realesrgan_only(input_path: Path, output_dir: Path) -> Path | None:
    """Real-ESRGAN ×4 без восстановления лиц."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REALESRGAN_SCRIPT),
        "-n", "RealESRGAN_x4plus",
        "--model_path", str(REALESRGAN_MODEL),
        "-i", str(input_path),
        "-o", str(output_dir),
        "--outscale", "4",
        "--tile", "512",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.stdout:
        log.info("ESRGAN: %s", r.stdout[-500:])
    if r.returncode != 0:
        log.error("ESRGAN завершился с ошибкой: %s", r.stderr[-500:])
        return None

    results = sorted(output_dir.glob("*.*"))
    if not results:
        log.error("ESRGAN: файл результата не найден")
        return None

    log.info("ESRGAN готово: %s", results[0].name)
    return results[0]


def _run_codeformer_only(input_path: Path, output_dir: Path, fidelity: float = 0.5) -> Path | None:
    """Только CodeFormer — восстановление лиц без апскейла фона."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_codeformer_weights()

    cmd = [
        sys.executable, str(CODEFORMER_SCRIPT),
        "-w", str(fidelity),
        "-s", "1",
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--face_upsample",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd="/app")
    if r.stdout:
        log.info("CodeFormer: %s", r.stdout[-500:])
    if r.returncode != 0:
        log.error("CodeFormer завершился с ошибкой: %s", r.stderr[-500:])
        return None

    final_dir = output_dir / "final_results"
    results = sorted(final_dir.glob("*.*")) if final_dir.exists() else []
    if not results:
        log.error("CodeFormer: файл результата не найден в final_results/")
        return None

    log.info("CodeFormer готово: %s", results[0].name)
    return results[0]


def _save_jpeg(src: Path, dst: Path) -> None:
    from PIL import Image
    img = Image.open(src).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_TELEGRAM_PX:
        scale = MAX_TELEGRAM_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        log.info("Resized output to %dx%d", img.size[0], img.size[1])
    img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def enhance(input_path: str, action: str = "enhance") -> str:
    """
    action:
        enhance    — CodeFormer (w=0.5, s=2) + апскейл фона Real-ESRGAN
        esrgan     — только Real-ESRGAN ×4
        codeformer — только CodeFormer, без апскейла фона
    """
    task_id = uuid.uuid4().hex
    work_dir = Path(f"/tmp/enhance_{task_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_BASE / f"enhance_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(input_path)
    h, w = _image_size(src)
    log.info("Enhance [%s] start: %s [%dx%d]", action, src.name, w, h)

    if action == "enhance":
        result = _run_enhance(src, work_dir / "out", fidelity=0.5, upscale=2)
    elif action == "esrgan":
        result = _run_realesrgan_only(src, work_dir / "out")
    elif action == "codeformer":
        result = _run_codeformer_only(src, work_dir / "out")
    else:
        result = _run_enhance(src, work_dir / "out", fidelity=0.5, upscale=2)

    if result is None:
        # если все движки упали — отдаём оригинал чтобы не подвешивать пользователя
        log.warning("Все движки упали, возвращаем оригинал")
        result = src

    final = out_dir / (src.stem + "_enhanced.jpg")
    _save_jpeg(result, final)
    h2, w2 = _image_size(final)
    log.info("Enhance done: %dx%d -> %dx%d | %s", w, h, w2, h2, final.name)
    return str(final)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    img = Image.open(path)
    return img.size[1], img.size[0]  # h, w
