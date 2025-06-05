import json
import os
import logging
import asyncio
from io import BytesIO
from random import choice
from gtts import gTTS
from telegram import BotCommand
from telegram import InputFile

from threading import Thread

from pydub import AudioSegment
import pyttsx3
from vosk import Model, KaldiRecognizer

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from dialogues import Dialogues

# Настройки
BOT_TOKEN = "7776222304:AAFYnO_1P46B7F9O4W2MpCiZBT3crivoNjs"
VOSK_MODEL_PATH = "models/"  # Путь к модели Vosk

# Инициализация синтезатора речи (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 50)  # Скорость речи
engine.setProperty('voice', 'ru')  # Русский голос (должен быть установлен в системе)


class Bot:
    def __init__(self, dialogues: Dialogues) -> None:
        self._dialogues = dialogues
        self._event_loop = None
        self._application = None
        self.voice_enabled = {}

    def start(self) -> None:
        """Запуск бота"""
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                logging.info("Stop previous cycle before running")
                loop.stop()
        except Exception as e:
            logging.warning(f"Err during stop: {str(e)}")

        logging.info("New event cycle")
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        # Создание бота
        builder = ApplicationBuilder().token(BOT_TOKEN)
        self._application = builder.build()

        # Обработчики
        self._application.add_handler(CommandHandler("start", self._bot_callback_start))
        self._application.add_handler(CommandHandler("enable_voice", self._bot_enable_voice))
        self._application.add_handler(CommandHandler("disable_voice", self._bot_disable_voice))
        self._application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, self._bot_callback_message_or_voice))

        # Запуск бота
        logging.info("starting bot polling")
        self._application.run_polling(close_loop=True)

    async def _bot_callback_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка команды /start"""
        chat_id = update.effective_chat.id

        commands = [
            BotCommand("start", "Перезапустить бота"),
            BotCommand("enable_voice", "Включить голосовые ответы"),
            BotCommand("disable_voice", "Отключить голосовые ответы"),
        ]
        await self._application.bot.set_my_commands(commands)
        welcome_text = (
            "Привет! 😊 Я виртуальный помощник Домовёнок.\n"
            "Давай поболтаем! Чем могу помочь?\n"
            "Могу показать диваны, кровати, шкафы и другую мебель."
        )
        await context.bot.send_message(chat_id=chat_id, text=welcome_text)

    async def _bot_callback_message_or_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка текстовых и голосовых сообщений с случайным выбором способа ответа"""
        chat_id = update.effective_chat.id

        # Установка команд
        commands = [
            BotCommand("start", "Перезапустить бота"),
            BotCommand("enable_voice", "Включить голосовые ответы"),
            BotCommand("disable_voice", "Отключить голосовые ответы"),
        ]
        await context.bot.set_my_commands(commands)
        request_message = self._extract_text(update, context)

        # Если сообщение текстовое
        if request_message:
            try:
                # Пробуем распознать текст из сообщения
                text = request_message.strip()

                if not text:
                    return

                # Получаем ответ от бота
                responses = self._dialogues.next_message(text, chat_id)
                current_intent_name = self._dialogues.intent_predict(text)
                intent_data = self._dialogues.bot_config.get(current_intent_name, {})
                logging.info(f"intent_data: {intent_data}")

                user_topic = "any"  # или другой topic, если есть контекст

                # Используем get_pic для получения пути к картинке
                image_path = self._dialogues.get_pic(current_intent_name, user_topic)
                if image_path:
                    abs_path = os.path.abspath(image_path)
                    logging.info(f"File exists: {os.path.exists(abs_path)}, size: {os.path.getsize(abs_path)} bytes")
                    logging.info(f"Found image: {image_path}")
                    try:
                        with open(abs_path, "rb") as photo_file:
                            await context.bot.send_photo(chat_id=chat_id, photo=photo_file)
                    except Exception as e:
                        logging.error(f"Error sending image: {str(e)}")
                else:
                    logging.info(f"No images found for intent '{current_intent_name}' and topic '{user_topic}'")

                # Случайным образом выбираем способ ответа: текстовый или голосовой
                if self.voice_enabled.get(chat_id, True):  # по умолчанию голос включён
                    answer_type = choice(["text", "voice"])
                else:
                    answer_type = "text"

                if answer_type == "text":
                    # Ответ текстом
                    for response in responses:
                        await context.bot.send_message(chat_id=chat_id, text=response)
                else:
                    # Ответ голосом
                    for response in responses:
                        voice_data = self._local_text_to_speech(response)
                        await context.bot.send_voice(chat_id=chat_id, voice=voice_data)

            except Exception as e:
                logging.error(f"Error in message processing: {str(e)}")
                await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при обработке сообщения.")

        # Если сообщение голосовое
        elif update.message.voice:
            try:
                # Получаем голосовое сообщение
                voice = update.message.voice
                voice_file = await context.bot.get_file(voice.file_id)
                ogg_data = await voice_file.download_as_bytearray()
                audio = AudioSegment.from_ogg(BytesIO(ogg_data))
                wav_data = audio.export(format="wav").read()

                # Преобразуем голос в текст
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, self._dialogues.voice_to_text, wav_data)

                if not text:
                    await context.bot.send_message(chat_id=chat_id, text="Не удалось распознать голосовое сообщение")
                    return

                # Получаем ответ от бота
                responses = self._dialogues.next_message(text, chat_id)

                # Случайным образом выбираем способ ответа: текстовый или голосовой
                if self.voice_enabled.get(chat_id, True):  # по умолчанию голос включён
                    answer_type = choice(["text", "voice"])
                else:
                    answer_type = "text"

                if answer_type == "text":
                    # Ответ текстом
                    for response in responses:
                        await context.bot.send_message(chat_id=chat_id, text=response)
                else:
                    # Ответ голосом
                    for response in responses:
                        voice_data = self._local_text_to_speech(response)
                        await context.bot.send_voice(chat_id=chat_id, voice=voice_data)

            except Exception as e:
                logging.error(f"Voice recognition error: {str(e)}")
                await context.bot.send_message(chat_id=chat_id, text="Ошибка обработки голосового сообщения")

    async def _bot_enable_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self.voice_enabled[chat_id] = True
        await context.bot.send_message(chat_id=chat_id, text="Голосовой ответ включён ✅")

    async def _bot_disable_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self.voice_enabled[chat_id] = False
        await context.bot.send_message(chat_id=chat_id, text="Голосовой ответ отключён ❌")

    def _extract_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Извлекает текст из сообщения"""
        if update.message.caption:
            return update.message.caption.strip()
        elif context.args is not None:
            return str(" ".join(context.args)).strip()
        elif update.message.text:
            return update.message.text.strip()
        return ""

    def _local_text_to_speech(self, text: str):
        """Синтез речи через gTTS (Google Text-to-Speech)"""
        try:
            tts = gTTS(text, lang='ru')  # Указываем язык как русский
            output_path = "temp_voice.mp3"
            tts.save(output_path)

            # Читаем файл и отправляем
            with open(output_path, "rb") as f:
                voice_data = BytesIO(f.read())
                voice_data.name = "response.mp3"

            # Удаляем временный файл
            os.remove(output_path)

            return voice_data
        except Exception as e:
            logging.error(f"Error in text-to-speech conversion: {str(e)}")
            return None
