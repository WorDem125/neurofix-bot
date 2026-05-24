import sys
import uuid
import shutil
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

REALESRGAN_SCRIPT = Path("/app/engines/realesrgan/inference_realesrgan.py")
CODEFORMER_SCRIPT = Path("/app/engines/codeformer/inference_codeformer.py")
REALESRGAN_MODEL  = Path("/models/realesrgan/RealESRGAN_x4plus.pth")
CODEFORMER_MODEL  = Path("/models/codeformer/codeformer.pth")
OUTPUT_BASE = Path("/data/results")


def _run_realesrgan(input_path: Path, output_dir: Path) -> Path | None:
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


def _run_codeformer(input_path: Path, output_dir: Path, fidelity: float = 0.7) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CodeFormer ищет веса в weights/CodeFormer/ относительно своей директории
    weights_dir = CODEFORMER_SCRIPT.parent / "weights" / "CodeFormer"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_link = weights_dir / "codeformer.pth"
    if not weight_link.exists():
        weight_link.symlink_to(CODEFORMER_MODEL)

    cmd = [
        sys.executable, str(CODEFORMER_SCRIPT),
        "-w", str(fidelity),
        "-s", "1",
        "--bg_upsampler", "realesrgan",
        "--bg_tile", "400",
        "--input_path", str(input_path),
        "--output_path", str(output_dir),
        "--face_upsample",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.stdout:
        log.info("CodeFormer: %s", r.stdout[-500:])
    if r.returncode != 0:
        log.error("CodeFormer завершился с ошибкой: %s", r.stderr[-500:])
        return None

    # CodeFormer пишет результат в output_path/final_results/
    final_dir = output_dir / "final_results"
    results = sorted(final_dir.glob("*.*")) if final_dir.exists() else []
    if not results:
        log.error("CodeFormer: файл результата не найден в final_results/")
        return None

    log.info("CodeFormer готово: %s", results[0].name)
    return results[0]


def enhance(input_path: str, action: str = "enhance") -> str:
    """
    action:
        enhance    — ESRGAN × 4  →  CodeFormer (лица)
        esrgan     — только ESRGAN × 4
        codeformer — только CodeFormer (лица)
    """
    task_id = uuid.uuid4().hex
    work_dir = Path(f"/tmp/enhance_{task_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_BASE / f"enhance_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    current = Path(input_path)
    h, w = _image_size(current)
    log.info("Enhance [%s] start: %s [%dx%d]", action, current.name, w, h)

    # Шаг 1: Real-ESRGAN
    if action in ("enhance", "esrgan"):
        result = _run_realesrgan(current, work_dir / "esrgan")
        if result:
            current = result
        else:
            log.warning("ESRGAN пропущен, продолжаем без upscale")

    # Шаг 2: CodeFormer
    if action in ("enhance", "codeformer"):
        result = _run_codeformer(current, work_dir / "codeformer")
        if result:
            current = result
        else:
            log.warning("CodeFormer пропущен, продолжаем без face enhancement")

    # Финальный результат
    final = out_dir / current.name
    shutil.copy2(current, final)
    h2, w2 = _image_size(final)
    log.info("Enhance готово: %dx%d -> %dx%d | %s", w, h, w2, h2, final.name)
    return str(final)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    img = Image.open(path)
    return img.size[1], img.size[0]  # h, w
