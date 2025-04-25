import logging
from pathlib import Path
from config.bot_config import BOT_CONFIG
from core.nlp_processor import NLProcessor
from core.voice_processor import VoiceProcessor
from core.bot_processor import BotCore
from bot_app import BotApp


def configure_logging():
    """Настройка логирования с фильтрацией ненужных логов"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Базовые настройки
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )

    # Основной логгер приложения
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Фильтр для Vosk и HTTPX логов
    class CustomFilter(logging.Filter):
        def filter(self, record):
            return not (
                    'VoskAPI' in record.name or
                    'httpx' in record.name or
                    'asyncio' in record.name
            )

    # Файловый обработчик
    file_handler = logging.FileHandler(
        log_dir / "bot.log",
        encoding='utf-8',
        mode='a'
    )
    file_handler.addFilter(CustomFilter())
    file_handler.setLevel(logging.DEBUG)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.addFilter(CustomFilter())
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Отключаем логирование для специфичных библиотек
    logging.getLogger("vosk").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def main():
    configure_logging()
    logger = logging.getLogger("Main")

    try:
        logger.info("="*50)
        logger.info("Инициализация системы...")

        # ============ Проверка конфигурации ============
        logger.debug("Проверка конфигурации...")
        if not BOT_CONFIG.get("token"):
            raise ValueError("Отсутствует токен бота в конфигурации")

        logger.debug("Конфигурация загружена успешно")
        logger.debug("Среда: %s", BOT_CONFIG.get("env", "production"))

        # ============ Тест NLP процессора ============
        logger.info("Инициализация NLP процессора...")
        nlp = NLProcessor()

        test_phrases = [
            ("диван", "ожидаемый интент: 'диваны'"),
            ("покажи диваны", "ожидаемый интент: 'диваны'"),
            ("фото дивана", "ожидаемый интент: 'диваны_фото' или 'unknown'"),
            ("привет", "ожидаемый интент: 'hello'")
        ]

        for phrase, description in test_phrases:
            result = nlp.predict(phrase)
            logger.debug(
                "Тест NLP: '%s' -> '%s' (%s)",
                phrase, result, description
            )

        # ============ Проверка голосового процессора ============
        logger.info("Инициализация голосового процессора...")
        voice = VoiceProcessor(model_path="/app/model/vosk-model-small-ru-0.22")
        logger.debug("Голосовая модель загружена")

        # ============ Инициализация ядра бота ============
        logger.info("Создание ядра бота...")
        bot_core = BotCore(nlp, voice)

        # Тест обработки текста
        test_responses = [
            ("диваны", "ожидается список диванов"),
            ("привет", "ожидается приветствие")
        ]

        for intent, description in test_responses:
            response = bot_core._get_response(intent)
            logger.debug(
                "Тест ответов: интент '%s' -> ответ: '%s' (%s)",
                intent, response, description
            )

        # ============ Запуск приложения ============
        logger.info("Инициализация Telegram приложения...")
        bot_app = BotApp(BOT_CONFIG["token"], bot_core)

        logger.info("="*50)
        logger.info("Бот успешно инициализирован. Запуск...")
        bot_app.run()

    except Exception as e:
        logger.critical(
            "Критическая ошибка при запуске: %s\n%s",
            str(e),
            "="*50,
            exc_info=True
        )
        raise SystemExit(1) from e

if __name__ == "__main__":
    main()