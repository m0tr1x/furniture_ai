from telegram import Update
from telegram.ext import CallbackContext
from core.nlp_processor import NLProcessor


class TextHandler:
    def __init__(self, nlp: NLProcessor):
        self.nlp = nlp

    async def handle(self, update: Update, context: CallbackContext):
        intent = self.nlp.predict(update.message.text)
        # Обработка интента и генерация ответа
        await update.message.reply_text(f"Распознан интент: {intent}")