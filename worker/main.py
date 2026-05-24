import os
import json
import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import redis
from dotenv import load_dotenv
from engines.restore import restore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
STREAM_TASKS = "photo:tasks"
STREAM_RESULTS = "photo:results"
CONSUMER_GROUP = "workers"
CONSUMER_NAME = "worker-1"
BLOCK_MS = 5000        # блокирующий read до 5 сек
RECONNECT_DELAY = 5

UPLOAD_DIR = Path("/tmp/worker_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ENGINES = {"old_photo": restore}


def get_redis():
    return redis.from_url(
        REDIS_URL,
        socket_connect_timeout=10,
        socket_timeout=10,
        socket_keepalive=True,
    )


def ensure_group(r: redis.Redis):
    """Создаёт consumer group если не существует."""
    try:
        r.xgroup_create(STREAM_TASKS, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' создана", CONSUMER_GROUP)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def process_image(task_id: str, task: dict) -> dict:
    action = task.get("action", "old_photo")
    logger = logging.LoggerAdapter(log, {"task_id": task_id})
    logger.info("Начинаю обработку action=%s chat_id=%s", action, task.get("chat_id"))

    engine = ENGINES.get(action)
    if engine is None:
        return {
            "task_id": task_id,
            "chat_id": task["chat_id"],
            "status": "failed",
            "error": f"Движок '{action}' не подключён.",
        }

    img_bytes = base64.b64decode(task["image_b64"])
    filename = task.get("filename", f"{task_id}.jpg")
    input_path = UPLOAD_DIR / filename
    input_path.write_bytes(img_bytes)

    try:
        result_path = engine(str(input_path))
        result_bytes = Path(result_path).read_bytes()
        result_b64 = base64.b64encode(result_bytes).decode()
        logger.info("Готово, результат %d байт", len(result_bytes))
        return {
            "task_id": task_id,
            "chat_id": task["chat_id"],
            "status": "ok",
            "image_b64": result_b64,
        }
    except Exception as e:
        logger.error("Ошибка pipeline: %s", e, exc_info=True)
        return {
            "task_id": task_id,
            "chat_id": task["chat_id"],
            "status": "failed",
            "error": str(e),
        }
    finally:
        input_path.unlink(missing_ok=True)


def main():
    log.info("Worker (Streams) запущен, REDIS_URL=%s", REDIS_URL)
    r = None

    while True:
        try:
            if r is None:
                r = get_redis()
                r.ping()
                ensure_group(r)
                log.info("Redis подключён")

            # Читаем новые сообщения ('>' = только непрочитанные)
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_TASKS: ">"},
                count=1,
                block=BLOCK_MS,
            )

            if not messages:
                continue

            stream_name, entries = messages[0]
            for msg_id, fields in entries:
                # Поля в Stream — bytes, декодируем
                task = {k.decode(): v.decode() for k, v in fields.items()}
                task_id = task.get("task_id", msg_id.decode())
                logger = logging.LoggerAdapter(log, {"task_id": task_id})

                try:
                    result = process_image(task_id, task)
                except Exception as e:
                    logger.error("Неожиданная ошибка: %s", e, exc_info=True)
                    result = {
                        "task_id": task_id,
                        "chat_id": task.get("chat_id", ""),
                        "status": "failed",
                        "error": str(e),
                    }

                # Пишем результат в стрим
                result_fields = {k: str(v) for k, v in result.items()}
                r.xadd(STREAM_RESULTS, result_fields)
                logger.info("Результат записан в %s", STREAM_RESULTS)

                # ACK только после успешной записи результата
                r.xack(STREAM_TASKS, CONSUMER_GROUP, msg_id)
                logger.info("ACK отправлен msg_id=%s", msg_id.decode())

        except (redis.exceptions.ConnectionError,
                redis.exceptions.TimeoutError) as e:
            log.warning("Redis ошибка: %s. Reconnect через %ds...", e, RECONNECT_DELAY)
            r = None
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            log.error("Неожиданная ошибка: %s", e, exc_info=True)
            r = None
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()
