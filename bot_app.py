from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import logging
from core.bot_processor import BotCore
from pathlib import Path


class BotApp:
    def __init__(self, token: str, bot_core: BotCore):
        self.core = bot_core
        self.logger = logging.getLogger(__name__)
        self.app = Application.builder().token(token).build()

        # Регистрация обработчиков
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self.handle_message))

    async def start(self, update: Update, context: CallbackContext):
        """Обработчик команды /start"""
        await update.message.reply_text("Привет! Я бот мебельного магазина. Чем могу помочь?")

    async def handle_message(self, update: Update, context: CallbackContext):
        """
        Универсальный обработчик для текста и голоса.
        Определяет тип сообщения и вызывает соответствующий метод BotCore.
        """
        try:
            if update.message.voice:
                # Обработка голоса
                voice_file = await update.message.voice.get_file()
                voice_path = "voice_message.ogg"
                await voice_file.download_to_drive(voice_path)
                response = self.core.process_voice(voice_path)
                # Удаляем временный файл
                Path(voice_path).unlink(missing_ok=True)
            else:
                # Обработка текста
                response = self.core.process_text(update.message.text)

            await update.message.reply_text(response)

        except Exception as e:
            self.logger.error(f"Ошибка обработки сообщения: {e}")
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте еще раз.")

    def run(self):
        """Запуск бота"""
        self.logger.info("Бот запущен")
        self.app.run_polling()