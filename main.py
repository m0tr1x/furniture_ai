import logging
from telegram.ext import Application
from config import bot_config
from handlers import TextHandler
from core import NLProcessor


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


async def post_init(app: Application):
    await app.bot.set_my_commands([
        ("start", "Начало работы"),
        ("help", "Помощь")
    ])


def main():
    setup_logging()

    nlp = NLProcessor()
    # Здесь должна быть загрузка данных для обучения NLP

    app = Application.builder().token(bot_config.TOKEN).post_init(post_init).build()

    text_handler = TextHandler(nlp)
    app.add_handler(MessageHandler(filters.TEXT, text_handler.handle))

    app.run_polling()


if __name__ == '__main__':
    main()