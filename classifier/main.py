import json
import logging
import os
import time

import redis
from dotenv import load_dotenv

from analyze import analyze_image

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger("classifier")

REDIS_URL       = os.getenv("REDIS_URL", "redis://redis:6379")
STREAM_TASKS    = "classify:tasks"
CONSUMER_GROUP  = "classifiers"
CONSUMER_NAME   = "classifier-1"
BLOCK_MS        = 5000
RECONNECT_DELAY = 5
RESULT_TTL      = 120  # секунды хранения результата в Redis


def get_redis() -> redis.Redis:
    return redis.from_url(
        REDIS_URL,
        socket_connect_timeout=10,
        socket_timeout=10,
        socket_keepalive=True,
    )


def ensure_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_TASKS, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' создана", CONSUMER_GROUP)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def process(task_id: str, task: dict) -> None:
    image_b64 = task.get("image_b64", "")
    log.info("Анализирую task_id=%s", task_id)

    result = analyze_image(image_b64)
    result["task_id"] = task_id

    # Все значения в JSON чтобы булевы не превращались в строку "False" (truthy!)
    payload = {k: json.dumps(v) for k, v in result.items()}

    result_key = f"classify:result:{task_id}"
    r_write = get_redis()
    pipe = r_write.pipeline()
    pipe.hset(result_key, mapping=payload)
    pipe.expire(result_key, RESULT_TTL)
    pipe.execute()
    log.info("Результат записан → %s (valid=%s)", result_key, result.get("valid"))


def main() -> None:
    log.info("neurofix-classifier запущен, REDIS_URL=%s", REDIS_URL)
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
                    process(task_id, task)
                except Exception as e:
                    log.error("Ошибка обработки task_id=%s: %s", task_id, e, exc_info=True)
                    # Пишем ошибку чтобы бот не завис в ожидании
                    result_key = f"classify:result:{task_id}"
                    err_r = get_redis()
                    err_r.hset(result_key, mapping={
                        "task_id": task_id,
                        "valid": "False",
                        "error_type": "internal_error",
                        "description": "Внутренняя ошибка анализа.",
                    })
                    err_r.expire(result_key, RESULT_TTL)
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
