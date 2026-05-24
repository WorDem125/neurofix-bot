import json
import logging
import os
import time
from datetime import datetime, timezone

import redis
from dotenv import load_dotenv

from colorize import colorize_image

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger("colorizer")

REDIS_URL       = os.getenv("REDIS_URL", "redis://redis:6379")
STREAM_TASKS    = "colorize:tasks"
STREAM_RESULTS  = "photo:results"
CONSUMER_GROUP  = "colorizers"
CONSUMER_NAME   = "colorizer-1"
BLOCK_MS        = 5000
RECONNECT_DELAY = 5


def get_redis() -> redis.Redis:
    return redis.from_url(
        REDIS_URL,
        socket_connect_timeout=10,
        socket_timeout=600,
        socket_keepalive=True,
    )


def ensure_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_TASKS, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' создана", CONSUMER_GROUP)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def process(r: redis.Redis, task_id: str, task: dict) -> None:
    chat_id  = task.get("chat_id", "0")
    filename = task.get("filename", f"{task_id}.jpg")
    image_b64 = task.get("image_b64", "")

    log.info("Раскрашиваю task_id=%s chat_id=%s", task_id, chat_id)
    result = colorize_image(image_b64)

    if result["ok"]:
        r.xadd(STREAM_RESULTS, {
            "task_id":   task_id,
            "chat_id":   chat_id,
            "action":    "colorize",
            "status":    "ok",
            "image_b64": result["image_b64"],
            "filename":  filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        log.info("Результат записан в %s task_id=%s", STREAM_RESULTS, task_id)
    else:
        r.xadd(STREAM_RESULTS, {
            "task_id": task_id,
            "chat_id": chat_id,
            "action":  "colorize",
            "status":  "failed",
            "error":   result["error"],
        })
        log.error("Ошибка task_id=%s: %s", task_id, result["error"])


def main() -> None:
    log.info("neurofix-colorizer запущен, REDIS_URL=%s", REDIS_URL)
    r = None

    while True:
        try:
            if r is None:
                r = get_redis()
                r.ping()
                ensure_group(r)
                log.info("Redis подключён")

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
                task = {k.decode(): v.decode() for k, v in fields.items()}
                task_id = task.get("task_id", msg_id.decode())
                try:
                    process(r, task_id, task)
                except Exception as e:
                    log.error("Ошибка task_id=%s: %s", task_id, e, exc_info=True)
                    try:
                        r.xadd(STREAM_RESULTS, {
                            "task_id": task_id,
                            "chat_id": task.get("chat_id", "0"),
                            "action":  "colorize",
                            "status":  "failed",
                            "error":   "Внутренняя ошибка раскраски.",
                        })
                    except Exception:
                        pass
                finally:
                    r.xack(STREAM_TASKS, CONSUMER_GROUP, msg_id)

        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            log.warning("Redis ошибка: %s. Reconnect через %ds...", e, RECONNECT_DELAY)
            r = None
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            log.error("Неожиданная ошибка: %s", e, exc_info=True)
            r = None
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()
