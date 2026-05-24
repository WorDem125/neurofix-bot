import os
import sys
import base64
import asyncio
import logging
import tempfile
from pathlib import Path

import redis.asyncio as aioredis
from dotenv import load_dotenv

from engines.enhance import enhance

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("enhancer")

REDIS_URL  = os.getenv("REDIS_URL", "redis://redis:6379")
STREAM_IN  = "enhance:tasks"
STREAM_OUT = "photo:results"
GROUP      = "enhancers"
CONSUMER   = "enhancer-1"
BLOCK_MS   = 5000
MAX_ERRORS = 3


async def ensure_group(r: aioredis.Redis) -> None:
    try:
        await r.xgroup_create(STREAM_IN, GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' создана", GROUP)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def process(r: aioredis.Redis, msg_id: str, fields: dict) -> None:
    task_id   = fields.get("task_id", "unknown")
    chat_id   = fields.get("chat_id", "")
    image_b64 = fields.get("image_b64", "")
    action    = fields.get("action", "enhance")
    filename  = fields.get("filename", "photo.jpg")

    log.info("[%s] task_id=%s action=%s", msg_id, task_id, action)

    suffix = Path(filename).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    input_path = tmp.name
    tmp.close()

    result_path = None
    try:
        # Декодируем base64 → файл
        img_bytes = base64.b64decode(image_b64)
        Path(input_path).write_bytes(img_bytes)
        log.info("Фото сохранено: %s (%d байт)", input_path, len(img_bytes))

        result_path = enhance(input_path, action=action)
        log.info("Готово: %s", result_path)

        # Читаем результат → base64
        result_bytes = Path(result_path).read_bytes()
        result_b64 = base64.b64encode(result_bytes).decode()

        await r.xadd(STREAM_OUT, {
            "task_id":   task_id,
            "chat_id":   chat_id,
            "action":    action,
            "image_b64": result_b64,
            "status":    "ok",
        })
        log.info("Результат отправлен в %s", STREAM_OUT)

    except Exception as exc:
        log.exception("Ошибка при обработке %s: %s", task_id, exc)
        await r.xadd(STREAM_OUT, {
            "task_id": task_id,
            "chat_id": chat_id,
            "action":  action,
            "status":  "error",
            "error":   str(exc),
        })
    finally:
        Path(input_path).unlink(missing_ok=True)
        if result_path:
            Path(result_path).unlink(missing_ok=True)


async def main() -> None:
    log.info("neurofix-enhancer запускается, Redis: %s", REDIS_URL)
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)

    await ensure_group(r)
    log.info("Ожидание задач в потоке '%s'...", STREAM_IN)

    error_count = 0
    while True:
        try:
            msgs = await r.xreadgroup(
                GROUP, CONSUMER,
                {STREAM_IN: ">"},
                count=1,
                block=BLOCK_MS,
            )
            if not msgs:
                continue

            for _stream, entries in msgs:
                for msg_id, fields in entries:
                    try:
                        await process(r, msg_id, fields)
                    except Exception:
                        log.exception("Необработанная ошибка, msg_id=%s", msg_id)
                    finally:
                        await r.xack(STREAM_IN, GROUP, msg_id)

            error_count = 0

        except aioredis.RedisError as exc:
            error_count += 1
            log.error("Redis ошибка (%d/%d): %s", error_count, MAX_ERRORS, exc)
            if error_count >= MAX_ERRORS:
                log.critical("Превышен лимит ошибок Redis, выход")
                sys.exit(1)
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            break

    await r.aclose()
    log.info("Enhancer остановлен")


if __name__ == "__main__":
    asyncio.run(main())
