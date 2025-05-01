import json
import logging

from bot import Bot

from dialogues import Dialogues


def main() -> None:
    """main entrance"""
    logging.basicConfig(level=logging.INFO)

    # loading config
    logging.info("loading config from bot_config.json")
    with open("bot_config.json", "r", encoding="utf-8") as file:
        bot_config = json.load(file)

    # dialogues init
    dialogue = Dialogues(bot_config)

    # bot class init
    bot_handler = Bot(dialogue)

    # parse dialogues
    dialogue.parse_dialogues_from_file()

    # learning
    dialogue.train_classifier()

    # start
    bot_handler.start()


if __name__ == "__main__":
    main()