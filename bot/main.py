import asyncio
import io
import os
import json
import uuid
import base64
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
import redis.asyncio as aioredis

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

STREAM_TASKS    = "photo:tasks"
STREAM_ENHANCE  = "enhance:tasks"
STREAM_COLORIZE = "colorize:tasks"
STREAM_EMOTION  = "emotion:tasks"
STREAM_RESULTS  = "photo:results"
STREAM_CLASSIFY = "classify:tasks"
RESULTS_CONSUMER_GROUP = "bot-readers"
RESULTS_CONSUMER_NAME  = "bot-1"

CLASSIFY_TIMEOUT = 12  # секунд ожидания ответа классификатора

BTN_RESTORE  = "🔄 Реставрация фото"
BTN_ENHANCE  = "⚡ Улучшить качество"
BTN_COLORIZE = "🖌 Раскрасить фото"
BTN_EMOTION  = "👁 Определить эмоцию"
BTN_AUTO     = "⚙️ Автообработка"
BTN_ABOUT    = "◆ О проекте"

FEATURE_BUTTONS: set[str] = set()


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_ENHANCE), KeyboardButton(text=BTN_RESTORE)],
            [KeyboardButton(text=BTN_COLORIZE), KeyboardButton(text=BTN_AUTO)],
            [KeyboardButton(text=BTN_EMOTION), KeyboardButton(text=BTN_ABOUT)],
        ],
        resize_keyboard=True,
        input_field_placeholder="Отправьте фото для обработки",
    )


WELCOME = (
    "Привет! Я <b>NeuroFix</b> — нейросеть для восстановления и улучшения фотографий 👋\n\n"
    "Отправьте снимок — я проанализирую его и предложу подходящие действия:\n\n"
    "⚡ <b>Улучшить качество</b> — чёткость, детализация, устранение шумов\n"
    "🔄 <b>Реставрация</b> — восстановление старых и повреждённых снимков\n"
    "🎨 <b>Раскраска</b> — превращаю чёрно-белые фото в цветные\n"
    "👁 <b>Анализ портрета</b> — эмоции, возраст, пол\n\n"
    "Просто загрузите снимок — остальное сделаю сам."
)
RESTORE_STARTED   = "Принято ✅\n\nВосстанавливаю фотографию, это займёт около минуты ⏳"
ENHANCE_STARTED   = "Принято ✅\n\nУлучшаю качество фотографии, это займёт около минуты ⏳"
RESTORE_NO_PHOTO  = "Загрузите фотографию — я её обработаю."
RESTORE_DONE      = "✅ Реставрация завершена\n\nПрименяю улучшение качества... ⏳"
RESTORE_ENHANCE_DONE = (
    "Готово! 🎉\n\n"
    "Фотография отреставрирована и улучшена — "
    "устранены повреждения, повышены чёткость и детализация."
)
ENHANCE_DONE      = (
    "Готово! 🎉\n\n"
    "Качество улучшено — повышены чёткость и детализация."
)
COLORIZE_STARTED  = "Принято ✅\n\nРаскрашиваю фотографию, это займёт около минуты ⏳"
COLORIZE_DONE     = (
    "Готово! 🎉\n\n"
    "Чёрно-белый снимок раскрашен — цвета подобраны нейросетью на основе контекста и освещения."
)
EMOTION_STARTED   = "Принято ✅\n\nАнализирую портрет, подождите немного ⏳"
FEATURE_IN_DEV    = "Эта функция скоро появится 🛠\n\nСледите за обновлениями."
ABOUT = (
    "✨ <b>NeuroFix — AI для ваших фотографий</b>\n\n"
    "Восстанавливаем, улучшаем и анализируем снимки с помощью нейросетей. "
    "Быстро, точно — прямо в Telegram.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "📸 <b>Что умеем:</b>\n\n"
    "Старое фото с царапинами\n"
    "→ чистый восстановленный снимок\n\n"
    "Размытое или низкокачественное фото\n"
    "→ чёткость и детализация ×4\n\n"
    "Чёрно-белый архивный снимок\n"
    "→ живые, реалистичные цвета\n\n"
    "Портрет человека\n"
    "→ эмоции, возраст, пол, этническая группа\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "⚡ <b>Почему NeuroFix:</b>\n\n"
    "• Результат за секунды — обработка на GPU\n"
    "• Не нужны сторонние сайты и приложения\n"
    "• Сами определяем, что нужно снимку\n"
    "• Специализируемся на портретах и людях\n\n"
    "🔒 <b>Приватность:</b> фотографии не хранятся "
    "и не передаются третьим лицам."
)
UNKNOWN_MESSAGE = "Используйте кнопки меню или отправьте фотографию для обработки."
PHOTO_TOO_LARGE = "Файл слишком большой.\n\nМаксимальный размер — 10 МБ. Попробуйте сжать фото."
ANALYZING       = "⏳ Анализирую фото..."

LAST_USER_IMAGES: dict[int, str] = {}
LAST_CLASSIFY: dict[int, dict] = {}        # user_id → результат классификатора
_auto_enhance_tasks: set[str] = set()      # task_ids авто-улучшения после ручной реставрации
_auto_pipeline: dict[str, dict] = {}       # task_id → состояние авто-пайплайна

# утилиты авто-пайплайна

_ACTION_STREAM = {
    "old_photo": "photo:tasks",
    "enhance":   "enhance:tasks",
    "colorize":  "colorize:tasks",
}
_ACTION_EMOJI = {
    "old_photo": "🔄",
    "enhance":   "⚡",
    "colorize":  "🖌",
}
_ACTION_RU = {
    "old_photo": "Реставрация",
    "enhance":   "Улучшение качества",
    "colorize":  "Раскраска",
}
# Согласование глагола с родом существительного
_ACTION_DONE = {
    "old_photo": "завершена",
    "enhance":   "завершено",
    "colorize":  "завершена",
}


def _build_pipeline(is_bw: bool, has_damage: bool) -> list[str]:
    steps = []
    if has_damage:
        steps.append("old_photo")
    steps.append("enhance")
    if is_bw:
        steps.append("colorize")
    return steps


def _auto_start_text(pipeline: list[str], is_bw: bool, has_damage: bool) -> str:
    lines = ["⚙️ <b>Автообработка запущена</b>\n"]
    if has_damage:
        lines.append("• Обнаружены повреждения — выполню реставрацию")
    if is_bw:
        lines.append("• Чёрно-белое фото — раскрашу в финале")
    if not has_damage and not is_bw:
        lines.append("• Фото в хорошем состоянии — повышу качество")
    total = len(pipeline)
    chain = " → ".join(f"{_ACTION_EMOJI[a]} {_ACTION_RU[a]}" for a in pipeline)
    lines.append(f"\n<b>План ({total} шаг{'а' if total == 2 else 'ов' if total == 3 else ''}):</b>")
    lines.append(chain)
    lines.append(f"\n{_ACTION_EMOJI[pipeline[0]]} <b>Шаг 1 из {total}: {_ACTION_RU[pipeline[0]]}...</b>")
    return "\n".join(lines)


def _auto_step_done_caption(step: int, total: int, action: str, is_last: bool) -> str:
    name = _ACTION_RU[action]
    done = _ACTION_DONE[action]
    if is_last:
        return f"✅ Шаг {step} из {total}: {name} {done}"
    return f"✅ Шаг {step} из {total}: {name} {done}\nПерехожу к следующему шагу..."


def _auto_next_step_text(step: int, total: int, action: str) -> str:
    emoji = _ACTION_EMOJI[action]
    name  = _ACTION_RU[action]
    return f"{emoji} <b>Шаг {step} из {total}: {name}...</b>"


def _auto_final_caption(pipeline: list[str]) -> str:
    chain = " → ".join(f"{_ACTION_EMOJI[a]} {_ACTION_RU[a]}" for a in pipeline)
    return f"🎉 <b>Автообработка завершена!</b>\n\n{chain}"

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            REDIS_URL,
            socket_connect_timeout=10,
            socket_keepalive=True,
        )
    return _redis_client


async def ensure_results_group(r: aioredis.Redis):
    try:
        await r.xgroup_create(STREAM_RESULTS, RESULTS_CONSUMER_GROUP, id="$", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            log.warning("xgroup_create results: %s", e)


async def classify_photo(image_b64: str) -> dict | None:
    """Отправляет фото классификатору и ждёт результат до CLASSIFY_TIMEOUT сек."""
    task_id = str(uuid.uuid4())
    result_key = f"classify:result:{task_id}"

    try:
        r = await get_redis()
        await r.xadd(STREAM_CLASSIFY, {
            "task_id":   task_id,
            "image_b64": image_b64,
        }, maxlen=50, approximate=True)
        log.info("classify задача отправлена task_id=%s", task_id)

        # Поллим результат с таймаутом
        deadline = asyncio.get_event_loop().time() + CLASSIFY_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            raw = await r.hgetall(result_key)
            if raw:
                # Все поля сохранены через json.dumps — читаем через json.loads
                result = {}
                for k, v in raw.items():
                    key = k.decode()
                    try:
                        result[key] = json.loads(v.decode())
                    except Exception:
                        result[key] = v.decode()
                log.info("classify результат получен task_id=%s valid=%s is_bw=%s damage=%s",
                         task_id, result.get("valid"), result.get("is_bw"), result.get("has_damage"))
                return result
            await asyncio.sleep(0.4)

        log.warning("classify таймаут task_id=%s", task_id)
        return None

    except Exception as e:
        log.error("classify ошибка: %s", e, exc_info=True)
        return None


def _format_classify_message(result: dict) -> str:
    lines = ["✅ Фото проанализировано. Доступные действия:"]

    is_bw      = result.get("is_bw", False)
    has_damage = result.get("has_damage", False)

    if has_damage:
        lines.append("  • 🔄 Реставрация — убрать повреждения и царапины")
    lines.append("  • ⚡ Улучшить качество — чёткость и детали")
    if is_bw:
        lines.append("  • 🖌 Раскраска — сделать цветным")
    lines.append("")
    lines.append("⚙️ <b>Автообработка</b> — выполню всё автоматически")

    return "\n".join(lines)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(WELCOME, reply_markup=main_menu())


@dp.message(lambda m: m.photo is not None)
async def handle_photo(message: Message):
    user_id = message.from_user.id
    photo = message.photo[-1]

    if photo.file_size and photo.file_size > 10 * 1024 * 1024:
        await message.answer(PHOTO_TOO_LARGE, reply_markup=main_menu())
        return

    # Скачиваем фото
    try:
        file = await bot.get_file(photo.file_id)
        buf = await bot.download_file(file.file_path)
        data = buf.read()
        image_b64 = base64.b64encode(data).decode()
        log.info("user_id=%s загрузил фото %d байт", user_id, len(data))
    except Exception as e:
        log.error("Ошибка скачивания фото: %s", e, exc_info=True)
        await message.answer("Не удалось получить изображение. Попробуйте ещё раз.", reply_markup=main_menu())
        return

    # Сообщаем что анализируем
    status_msg = await message.answer(ANALYZING)

    # Отправляем на классификацию
    result = await classify_photo(image_b64)

    if result is None:
        # Классификатор не ответил — сохраняем фото без анализа
        LAST_USER_IMAGES[user_id] = image_b64
        await status_msg.edit_text("Фото загружено.\n\nВыберите действие для обработки.")
        return

    if not result.get("valid", False):
        # Невалидное фото — не сохраняем, объясняем причину
        error_text = result.get("description", "Изображение не подходит для обработки.")
        await status_msg.edit_text(f"❌ {error_text}")
        return

    # Валидное фото — сохраняем и показываем анализ
    LAST_USER_IMAGES[user_id] = image_b64
    LAST_CLASSIFY[user_id] = result
    reply_text = _format_classify_message(result)
    await status_msg.edit_text(reply_text)


@dp.message(lambda m: m.text == BTN_RESTORE)
async def handle_restore(message: Message):
    user_id = message.from_user.id
    image_b64 = LAST_USER_IMAGES.get(user_id)

    if not image_b64:
        await message.answer(RESTORE_NO_PHOTO, reply_markup=main_menu())
        return

    task_id = str(uuid.uuid4())
    await message.answer(RESTORE_STARTED, reply_markup=main_menu())

    try:
        r = await get_redis()
        msg_id = await r.xadd(STREAM_TASKS, {
            "task_id":    task_id,
            "chat_id":    str(message.chat.id),
            "action":     "old_photo",
            "filename":   f"{task_id}.jpg",
            "image_b64":  image_b64,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, maxlen=50, approximate=True)
        log.info("Restore задача добавлена msg_id=%s chat_id=%s", msg_id, message.chat.id)
    except Exception as e:
        log.error("Ошибка отправки в Redis Streams: %s", e, exc_info=True)
        await message.answer("Ошибка подключения к серверу обработки. Попробуйте позже.", reply_markup=main_menu())


@dp.message(lambda m: m.text == BTN_ENHANCE)
async def handle_enhance(message: Message):
    user_id = message.from_user.id
    image_b64 = LAST_USER_IMAGES.get(user_id)

    if not image_b64:
        await message.answer(RESTORE_NO_PHOTO, reply_markup=main_menu())
        return

    task_id = str(uuid.uuid4())
    await message.answer(ENHANCE_STARTED, reply_markup=main_menu())

    try:
        r = await get_redis()
        msg_id = await r.xadd(STREAM_ENHANCE, {
            "task_id":    task_id,
            "chat_id":    str(message.chat.id),
            "action":     "enhance",
            "filename":   f"{task_id}.jpg",
            "image_b64":  image_b64,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, maxlen=50, approximate=True)
        log.info("Enhance задача добавлена msg_id=%s chat_id=%s", msg_id, message.chat.id)
    except Exception as e:
        log.error("Ошибка отправки в Redis Streams: %s", e, exc_info=True)
        await message.answer("Ошибка подключения к серверу обработки. Попробуйте позже.", reply_markup=main_menu())


@dp.message(lambda m: m.text == BTN_COLORIZE)
async def handle_colorize(message: Message):
    user_id = message.from_user.id
    image_b64 = LAST_USER_IMAGES.get(user_id)

    if not image_b64:
        await message.answer(RESTORE_NO_PHOTO, reply_markup=main_menu())
        return

    task_id = str(uuid.uuid4())
    await message.answer(COLORIZE_STARTED, reply_markup=main_menu())

    try:
        r = await get_redis()
        msg_id = await r.xadd(STREAM_COLORIZE, {
            "task_id":    task_id,
            "chat_id":    str(message.chat.id),
            "action":     "colorize",
            "filename":   f"{task_id}.jpg",
            "image_b64":  image_b64,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, maxlen=50, approximate=True)
        log.info("Colorize задача добавлена msg_id=%s chat_id=%s", msg_id, message.chat.id)
    except Exception as e:
        log.error("Ошибка отправки в Redis Streams: %s", e, exc_info=True)
        await message.answer("Ошибка подключения к серверу обработки. Попробуйте позже.", reply_markup=main_menu())


@dp.message(lambda m: m.text == BTN_EMOTION)
async def handle_emotion(message: Message):
    user_id = message.from_user.id
    image_b64 = LAST_USER_IMAGES.get(user_id)

    if not image_b64:
        await message.answer(RESTORE_NO_PHOTO, reply_markup=main_menu())
        return

    task_id = str(uuid.uuid4())
    await message.answer(EMOTION_STARTED, reply_markup=main_menu())

    try:
        r = await get_redis()
        msg_id = await r.xadd(STREAM_EMOTION, {
            "task_id":    task_id,
            "chat_id":    str(message.chat.id),
            "image_b64":  image_b64,
            "filename":   f"{task_id}.jpg",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, maxlen=50, approximate=True)
        log.info("Emotion задача добавлена msg_id=%s chat_id=%s", msg_id, message.chat.id)
    except Exception as e:
        log.error("Ошибка отправки в Redis Streams: %s", e, exc_info=True)
        await message.answer("Ошибка подключения к серверу обработки. Попробуйте позже.", reply_markup=main_menu())


@dp.message(lambda m: m.text == BTN_AUTO)
async def handle_auto(message: Message):
    user_id = message.from_user.id
    image_b64 = LAST_USER_IMAGES.get(user_id)

    if not image_b64:
        await message.answer(RESTORE_NO_PHOTO, reply_markup=main_menu())
        return

    classify = LAST_CLASSIFY.get(user_id)
    if classify is None:
        # Нет результата классификации — запускаем заново
        status_msg = await message.answer(ANALYZING)
        classify = await classify_photo(image_b64)
        if classify is None or not classify.get("valid", False):
            await status_msg.edit_text(
                "⚙️ Не удалось проанализировать фото. "
                "Попробуйте загрузить снимок заново."
            )
            return
        LAST_CLASSIFY[user_id] = classify
        await status_msg.delete()

    is_bw      = classify.get("is_bw", False)
    has_damage = classify.get("has_damage", False)
    pipeline   = _build_pipeline(is_bw, has_damage)

    # Запускаем первый шаг
    task_id = str(uuid.uuid4())
    first_action = pipeline[0]

    _auto_pipeline[task_id] = {
        "chat_id":    message.chat.id,
        "pipeline":   pipeline,
        "step":       0,
        "is_bw":      is_bw,
        "has_damage": has_damage,
    }

    await message.answer(_auto_start_text(pipeline, is_bw, has_damage), reply_markup=main_menu())

    try:
        r = await get_redis()
        await r.xadd(_ACTION_STREAM[first_action], {
            "task_id":    task_id,
            "chat_id":    str(message.chat.id),
            "action":     first_action,
            "filename":   f"{task_id}.jpg",
            "image_b64":  image_b64,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        log.info("Auto-pipeline запущен task_id=%s action=%s chat_id=%s pipeline=%s",
                 task_id, first_action, message.chat.id, pipeline)
    except Exception as e:
        del _auto_pipeline[task_id]
        log.error("Ошибка запуска auto-pipeline: %s", e, exc_info=True)
        await message.answer("Ошибка подключения к серверу. Попробуйте позже.", reply_markup=main_menu())


@dp.message(lambda m: m.text in FEATURE_BUTTONS)
async def handle_feature(message: Message):
    await message.answer(FEATURE_IN_DEV, reply_markup=main_menu())


@dp.message(lambda m: m.text == BTN_ABOUT)
async def handle_about(message: Message):
    await message.answer(ABOUT, reply_markup=main_menu())


@dp.message()
async def handle_unknown(message: Message):
    await message.answer(UNKNOWN_MESSAGE, reply_markup=main_menu())


async def poll_results():
    log.info("poll_results (Streams) запущен")
    r = None
    last_id = "$"

    while True:
        try:
            if r is None:
                r = await aioredis.from_url(
                    REDIS_URL,
                    socket_connect_timeout=10,
                    socket_keepalive=True,
                )
                await ensure_results_group(r)
                log.info("poll_results: Redis подключён")

            results = await r.xread({STREAM_RESULTS: last_id}, count=10, block=3000)

            if not results:
                continue

            stream_name, entries = results[0]
            await r.xtrim(STREAM_RESULTS, maxlen=30, approximate=True)
            for msg_id, fields in entries:
                last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                result = {k.decode(): v.decode() for k, v in fields.items()}
                task_id = result.get("task_id", last_id)
                chat_id = int(result.get("chat_id", 0))
                status  = result.get("status", "failed")
                action  = result.get("action", "old_photo")

                log.info("Результат task_id=%s action=%s status=%s chat_id=%s",
                         task_id, action, status, chat_id)

                # авто-пайплайн: цепочка задач
                if task_id in _auto_pipeline:
                    state    = _auto_pipeline[task_id]
                    pipeline = state["pipeline"]
                    step_idx = state["step"]
                    total    = len(pipeline)

                    if status != "ok" or not result.get("image_b64"):
                        del _auto_pipeline[task_id]
                        err = result.get("error", "Произошла ошибка при автообработке.")
                        await bot.send_message(state["chat_id"], f"❌ {err}", reply_markup=main_menu())
                        log.warning("Auto-pipeline ошибка на шаге %d/%d task_id=%s", step_idx + 1, total, task_id)
                        continue

                    img_b64   = result["image_b64"]
                    img_bytes = base64.b64decode(img_b64)
                    LAST_USER_IMAGES[state["chat_id"]] = img_b64

                    is_last = (step_idx == total - 1)

                    if is_last:
                        del _auto_pipeline[task_id]
                        caption    = _auto_final_caption(pipeline)
                        photo_file = BufferedInputFile(img_bytes, filename="result.jpg")
                        await bot.send_photo(state["chat_id"], photo_file, caption=caption, reply_markup=main_menu())
                        log.info("Auto-pipeline завершён chat_id=%s pipeline=%s", state["chat_id"], pipeline)
                    else:
                        # Промежуточный шаг — отправляем фото, ставим следующий
                        caption    = _auto_step_done_caption(step_idx + 1, total, action, is_last=False)
                        photo_file = BufferedInputFile(img_bytes, filename="result.jpg")
                        await bot.send_photo(state["chat_id"], photo_file, caption=caption)

                        next_idx    = step_idx + 1
                        next_action = pipeline[next_idx]
                        next_tid    = str(uuid.uuid4())

                        _auto_pipeline[next_tid] = {
                            "chat_id":    state["chat_id"],
                            "pipeline":   pipeline,
                            "step":       next_idx,
                            "is_bw":      state["is_bw"],
                            "has_damage": state["has_damage"],
                        }
                        del _auto_pipeline[task_id]

                        try:
                            await r.xadd(_ACTION_STREAM[next_action], {
                                "task_id":    next_tid,
                                "chat_id":    str(state["chat_id"]),
                                "action":     next_action,
                                "filename":   f"{next_tid}.jpg",
                                "image_b64":  img_b64,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            }, maxlen=50, approximate=True)
                            await bot.send_message(
                                state["chat_id"],
                                _auto_next_step_text(next_idx + 1, total, next_action),
                            )
                            log.info("Auto-pipeline шаг %d/%d → %s next_tid=%s",
                                     step_idx + 1, total, next_action, next_tid)
                        except Exception as eq:
                            del _auto_pipeline[next_tid]
                            log.error("Ошибка постановки следующего шага auto-pipeline: %s", eq)
                            await bot.send_message(
                                state["chat_id"],
                                "❌ Ошибка при переходе к следующему шагу.",
                                reply_markup=main_menu(),
                            )
                    continue

                if status == "ok" and action == "emotion":
                    # Текстовый результат — анализ портрета
                    text = result.get("text", "Анализ завершён.")
                    await bot.send_message(chat_id, text, reply_markup=main_menu())
                    log.info("Emotion результат отправлен chat_id=%s", chat_id)

                elif status == "ok" and result.get("image_b64"):
                    img_b64 = result["image_b64"]
                    img_bytes = base64.b64decode(img_b64)
                    # Сохраняем ДО отправки — исключаем race condition
                    LAST_USER_IMAGES[chat_id] = img_b64

                    if action == "enhance" and task_id in _auto_enhance_tasks:
                        # Финал авто-пайплайна: отреставрировано + улучшено
                        _auto_enhance_tasks.discard(task_id)
                        caption = RESTORE_ENHANCE_DONE
                    elif action == "enhance":
                        caption = ENHANCE_DONE
                    elif action == "colorize":
                        caption = COLORIZE_DONE
                    else:
                        # Реставрация завершена → автоматически ставим на улучшение
                        caption = RESTORE_DONE
                        auto_task_id = str(uuid.uuid4())
                        _auto_enhance_tasks.add(auto_task_id)
                        try:
                            await r.xadd(STREAM_ENHANCE, {
                                "task_id":    auto_task_id,
                                "chat_id":    str(chat_id),
                                "action":     "enhance",
                                "filename":   f"{auto_task_id}.jpg",
                                "image_b64":  img_b64,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            }, maxlen=50, approximate=True)
                            log.info("Авто-улучшение поставлено в очередь auto_task_id=%s chat_id=%s",
                                     auto_task_id, chat_id)
                        except Exception as eq:
                            _auto_enhance_tasks.discard(auto_task_id)
                            log.error("Ошибка постановки авто-улучшения: %s", eq)

                    photo_file = BufferedInputFile(img_bytes, filename="result.jpg")
                    await bot.send_photo(chat_id, photo_file, caption=caption, reply_markup=main_menu())
                    log.info("Фото отправлено chat_id=%s action=%s", chat_id, action)
                else:
                    error_text = result.get("error", "Произошла ошибка при обработке.")
                    await bot.send_message(chat_id, f"❌ {error_text}", reply_markup=main_menu())
                    log.warning("Ошибка: %s", error_text)

        except Exception as e:
            log.error("poll_results ошибка: %s", e, exc_info=True)
            r = None
            await asyncio.sleep(5)


_poll_task: asyncio.Task | None = None


async def on_startup(bot: Bot):
    global _poll_task
    log.info("NeuroFix Bot запускается...")
    await bot.set_my_commands([
        BotCommand(command="start", description="Запустить бота"),
    ])
    _poll_task = asyncio.create_task(poll_results(), name="poll_results")


async def on_shutdown(bot: Bot):
    global _poll_task
    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass


dp.startup.register(on_startup)
dp.shutdown.register(on_shutdown)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
