import logging
import json
import random
from typing import Dict
from pathlib import Path


class BotCore:
    def __init__(self, nlp_processor, voice_processor):
        self.nlp = nlp_processor
        self.voice_processor = voice_processor
        self.logger = logging.getLogger(__name__)
        self.phrases = self._load_phrases()

    def _load_phrases(self) -> Dict:
        """Загрузка фраз из JSON-файла с резервными вариантами"""
        default_phrases = {
            "intents": {
                "default": ["Отличный выбор! Что вас интересует?"]
            },
            "errors": {
                "default": ["Произошла непредвиденная ошибка"],
                "voice": ["Не удалось обработать голосовое сообщение"]
            }
        }

        try:
            path = Path(__file__).parent.parent / 'data' / 'phrases.json'
            with open(path, 'r', encoding='utf-8') as f:
                user_phrases = json.load(f)
                # Объединяем с дефолтными фразами для защиты от отсутствующих ключей
                return {**default_phrases, **user_phrases}
        except Exception as e:
            self.logger.error(f"Ошибка загрузки phrases.json: {e}")
            return default_phrases


    def process_voice(self, voice_path: str) -> str:
        """Обработка голоса с учетом речевых ошибок"""
        try:
            recognized_text = self.voice_processor.recognize(voice_path)
            if not recognized_text or "Ошибка" in recognized_text:
                return random.choice(self.phrases["errors"]["voice"])

            intent = self.nlp.predict(recognized_text)
            return self._get_response(intent)
        except Exception as e:
            self.logger.error(f"Ошибка голоса: {e}")
            return random.choice(self.phrases["errors"]["voice"])

    def process_text(self, text: str) -> str:
        """Обработка текста с логированием интента"""
        try:
            intent = self.nlp.predict(text)
            self.logger.info(f"Распознан интент: '{intent}' для текста: '{text}'")

            # Специальная обработка похожих интентов
            if intent == "hello":
                if "вечер" in text.lower():
                    return "Добрый вечер! Чем могу помочь?"
                elif "день" in text.lower():
                    return "Добрый день! Выбираете мебель?"

            return self._get_response(intent)

        except Exception as e:
            self.logger.error(f"Ошибка: {e}")
            return random.choice(self.phrases["errors"]["default"])

    def _get_response(self, intent: str) -> str:
        """Усовершенствованный выбор ответа с интеллектуальным избеганием повторов"""
        # Получаем все возможные ответы для интента
        responses = self.phrases["intents"].get(intent, [])

        # Если нет ответов для интента, используем ответы по умолчанию
        if not responses:
            self.logger.warning(f"Нет фраз для интента: {intent}")
            responses = self.phrases["errors"]["default"]

        # Инициализируем историю ответов, если её нет
        if not hasattr(self, '_response_history'):
            self._response_history = {}

        # Инициализируем историю для текущего интента, если её нет
        if intent not in self._response_history:
            self._response_history[intent] = {
                'all_responses': responses.copy(),
                'used_responses': [],
                'last_response': None
            }

        # Получаем доступные (ещё не использованные) ответы
        available = [
            r for r in self._response_history[intent]['all_responses']
            if r not in self._response_history[intent]['used_responses']
        ]

        # Если все ответы уже использовались, сбрасываем историю
        if not available:
            self.logger.debug(f"Все ответы для интента '{intent}' были использованы, сбрасываем историю")
            self._response_history[intent]['used_responses'] = []
            available = self._response_history[intent]['all_responses'].copy()

        # Выбираем случайный ответ из доступных
        response = random.choice(available)

        # Обновляем историю
        self._response_history[intent]['used_responses'].append(response)
        self._response_history[intent]['last_response'] = response

        self.logger.debug(f"Для интента '{intent}' выбран ответ: '{response}'")
        self.logger.debug(f"Осталось доступных ответов: {len(available) - 1}")

        return response