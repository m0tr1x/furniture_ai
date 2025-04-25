import json
from pathlib import Path
import os


def load_phrases_config():
    # Получаем абсолютный путь к директории config
    config_dir = Path(__file__).parent

    # Поднимаемся на уровень выше (в корень проекта) и идем в data/phrases.json
    phrases_path = config_dir.parent / 'data' / 'phrases.json'

    try:
        with open(phrases_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

            # Проверяем обязательные ключи
            if 'intents' not in config:
                raise ValueError("Конфиг должен содержать ключ 'intents'")

            return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл конфигурации не найден по пути: {phrases_path}")
    except json.JSONDecodeError:
        raise ValueError("Ошибка в формате JSON файла конфигурации")


# Загружаем конфиг при импорте модуля
BOT_CONFIG = load_phrases_config()