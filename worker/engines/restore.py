import os
import sys
import uuid
import shutil
import logging
import subprocess
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

ENGINE_DIR = Path(__file__).parent / "old_photo"
OUTPUT_BASE = Path("/data/results")
WORK_BASE = Path("/tmp/restore_work")

MIN_SIDE = 800   # upscale маленькие фото до этого минимума
MAX_SIDE = 1536  # ограничение GPU


def _prepare_image(src: Path, dst: Path) -> tuple[int, int]:
    """Нормализует размер изображения, возвращает оригинальные размеры."""
    img = Image.open(src).convert("RGB")
    orig_w, orig_h = img.size
    w, h = orig_w, orig_h

    longest = max(w, h)
    if longest < MIN_SIDE:
        scale = MIN_SIDE / longest
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)
        log.info("Upscaled %dx%d -> %dx%d", orig_w, orig_h, w, h)
    elif longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)
        log.info("Downscaled %dx%d -> %dx%d", orig_w, orig_h, w, h)

    img.save(dst, quality=95)
    return orig_w, orig_h


def _upscale_result(result_path: Path, orig_w: int, orig_h: int) -> Path:
    """Апскейл результата: минимум 1024px по длинной стороне."""
    img = Image.open(result_path).convert("RGB")
    w, h = img.size
    target = max(1024, max(orig_w, orig_h))
    target = min(target, MAX_SIDE)

    if max(w, h) < target:
        scale = target / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out = result_path.parent / ("up_" + result_path.name)
        img.save(out, quality=95)
        log.info("Final upscale %dx%d -> %dx%d", w, h, new_w, new_h)
        return out

    return result_path


def restore(input_path: str) -> str:
    input_path = Path(input_path)
    task_id = uuid.uuid4().hex

    work_dir = WORK_BASE / task_id / "input"
    work_dir.mkdir(parents=True, exist_ok=True)

    dest_name = input_path.stem + ".jpg"
    orig_w, orig_h = _prepare_image(input_path, work_dir / dest_name)

    output_dir = OUTPUT_BASE / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "run.py",
        "--input_folder", str(work_dir),
        "--output_folder", str(output_dir),
        "--GPU", "0",
        "--with_scratch",
        "--HR",
    ]

    log.info("Запуск restore (with_scratch + HR): %s [%dx%d]", input_path.name, orig_w, orig_h)
    result = subprocess.run(
        cmd,
        cwd=str(ENGINE_DIR),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    if result.stdout:
        log.info(result.stdout[-2000:])
    if result.stderr:
        log.warning(result.stderr[-2000:])

    if result.returncode != 0:
        raise RuntimeError(f"Движок завершился с кодом: {result.returncode}")

    for stage in ["final_output", "stage_3_face_restore_output", "stage_2_wholistic_output", "stage_1_restore_output"]:
        stage_dir = output_dir / stage
        if stage_dir.exists():
            results = sorted(stage_dir.rglob("*.png")) + sorted(stage_dir.rglob("*.jpg"))
            if results:
                best = results[0]
                final = _upscale_result(best, orig_w, orig_h)
                log.info("Результат: %s", final)
                return str(final)

    raise RuntimeError("Результат не найден в выходной папке")
