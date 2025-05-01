import json
import os
import logging
import asyncio
from io import BytesIO
from random import choice
from gtts import gTTS

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
        self._application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, self._bot_callback_message_or_voice))

        # Запуск бота
        logging.info("starting bot polling")
        self._application.run_polling(close_loop=True)

    async def _bot_callback_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка команды /start"""
        chat_id = update.effective_chat.id
        responses = self._dialogues.next_message("start", chat_id)
        for response in responses:
            await context.bot.send_message(chat_id=chat_id, text=response)

    async def _bot_callback_message_or_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка текстовых и голосовых сообщений с случайным выбором способа ответа"""
        chat_id = update.effective_chat.id
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

                # Случайным образом выбираем способ ответа: текстовый или голосовой
                answer_type = choice(["text", "voice"])

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
                answer_type = choice(["text", "voice"])

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
