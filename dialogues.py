import logging
import random
from typing import Dict, List, Union, Optional
import json
from io import BytesIO
import os
import vosk
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment

# Probability of ad
AD_PROBABILITY = 0.2

# Threshold for similarity in dialogues
DIALOGUES_THRESHOLD = 0.5

# symbols to confirm
ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890- "


class Dialogues:
    def __init__(self, bot_config: Dict) -> None:
        self.bot_config = bot_config
        self._dialogues_structured = {}
        self._vectorizer = None
        self._classifier = None
        self._users_topics = {}

        # Инициализация модели Vosk
        try:
            model_path = "/app/models"  # Путь к модели Vosk
            if not os.path.exists(model_path):
                logging.error(f"Vosk model not found at {model_path}")
                raise FileNotFoundError(f"Vosk model not found at {model_path}")

            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)  # Устанавливаем частоту дискретизации 16kHz
            logging.info("Vosk model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Vosk model: {str(e)}")
            self.model = None
            self.recognizer = None

    def voice_to_text(self, voice_data: bytes) -> Optional[str]:
        """Конвертация голосового сообщения в текст с использованием Vosk"""
        if not self.model:
            logging.error("Vosk model not initialized")
            return None

        try:
            logging.info("Starting voice recognition using Vosk")

            # Сохраняем голосовое сообщение во временный файл
            with open("temp_input.ogg", "wb") as f:
                f.write(voice_data)
            logging.info("Voice data written to temp_input.ogg")

            # Загрузка аудио с помощью pydub
            audio = AudioSegment.from_file("temp_input.ogg")
            logging.info(f"Audio file loaded successfully with duration: {len(audio)} ms")

            # Гарантированно преобразуем в моно 16kHz 16-bit PCM WAV
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export("temp_audio.wav", format="wav")
            logging.info("Audio converted to 16kHz mono 16-bit PCM WAV format")

            # 🟢 Создаём новый recognizer на каждый вызов!
            recognizer = vosk.KaldiRecognizer(self.model, 16000)
            logging.info(f"Created new recognizer object: {recognizer}")

            with open("temp_audio.wav", "rb") as f:
                while True:
                    data = f.read(4000)
                    if len(data) == 0:
                        break
                    if not recognizer.AcceptWaveform(data):
                        partial = recognizer.PartialResult()
                        logging.debug(f"Partial: {partial}")

                result = recognizer.FinalResult()
                logging.info(f"Final Vosk result raw: {result}")
                result_json = json.loads(result)
                text = result_json.get("text", "").strip()

                if text:
                    logging.info(f"Recognized text: {text}")
                    return text
                else:
                    # Попробуем взять хотя бы partial
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        logging.info(f"Recognized partial text: {partial}")
                        return partial
                    else:
                        logging.warning("Speech not recognized by Vosk (even partial)")
                        return None

        except Exception as e:
            logging.error(f"Voice recognition error: {str(e)}")
            return None

    def phrase_simplify(self, phrase: str) -> str:
        """Приведение фразы к нижнему регистру и удаление лишних символов"""
        return "".join(symbol for symbol in phrase.lower() if symbol in ALPHABET).strip()

    def next_message(
            self,
            request: Union[str, bytes],
            user_id: int,
            is_voice: bool = False
    ) -> List[str]:
        """
        Генерация следующего сообщения
        Возвращает список текстовых ответов
        """
        try:
            # Если это голосовое сообщение - конвертируем в текст
            if is_voice:
                if not isinstance(request, bytes):
                    logging.error("Voice message should be bytes")
                    return [random.choice(self.bot_config["failure"])]

                text = self.voice_to_text(request)
                if not text:
                    return ["Не удалось распознать голосовое сообщение"]
                request = text

            # Остальная логика обработки сообщения (как в оригинале)
            if user_id not in self._users_topics:
                logging.info(f"new user creation: {user_id}")
                self._users_topics[user_id] = "any"

            user_topic = self._users_topics[user_id]
            user_intent = self.intent_predict(request)

            if not user_intent or user_intent not in self.bot_config["intents"]:
                answer_from_dialogues = self.generate_answer_dialogues(request)
                if answer_from_dialogues:
                    return [answer_from_dialogues]
                return [random.choice(self.bot_config["failure"])]

            ad = AD_PROBABILITY > random.random() and user_intent in self.bot_config["ad_intents"]

            if user_topic in self.bot_config["intents"][user_intent]:
                topic = self.bot_config["intents"][user_intent][user_topic]
            else:
                topic = self.bot_config["intents"][user_intent]["any"]

            responses = [random.choice(topic["responses"])]

            if ad:
                responses.append(random.choice(self.bot_config["ad_reponses"]))

            return responses

        except Exception as e:
            logging.error(f'Error in next_message: {str(e)}')
            return [random.choice(self.bot_config["failure"])]


    def parse_dialogues_from_file(self) -> None:
        """load and parse dialogues.txt """

        logging.warning("loading from dialogues.txt")
        with open("dialogues.txt", "r", encoding="utf-8") as file:
            content = file.read()

        dialogues = [dialogue.split("\n")[:2] for dialogue in content.split("\n\n") if len(dialogue.split("\n")) == 2]

        logging.info("Filtering dialogues")
        dialogues_filtered = []
        questions = set()
        for dialogue in dialogues:
            question, answer = dialogue
            question = self.phrase_simplify(question[2:])
            answer = answer[2:]
            if question and question not in questions:
                questions.add(question)
                dialogues_filtered.append([question, answer])

        logging.info("Structuring")
        dialogues_structured = {}
        for question, answer in dialogues_filtered:
            words = set(question.split())
            for word in words:
                dialogues_structured.setdefault(word, []).append([question, answer])

        logging.info("Dialogs sorting")
        self._dialogues_structured = {
            word: sorted(pairs, key=lambda pair: len(pair[0]))[:1000] for word, pairs in dialogues_structured.items()
        }

        # Перемешать
        logging.info("Dialog mix")
        dialogues_structured_list = list(self._dialogues_structured.items())
        random.shuffle(dialogues_structured_list)
        self._dialogues_structured = dict(dialogues_structured_list)

        # Готово
        logging.info(f"Dialogues for {len(self._dialogues_structured)} words are loaded.")

    def train_classifier(self) -> None:
        """Creates and study classificator"""
        intent_names = []
        intent_examples = []

        for intent, dialogues_list in self._dialogues_structured.items():
            if len(intent_names) > 10000:
                break
            for dialogue_ in dialogues_list:
                intent_names.append(intent)
                intent_examples.append(dialogue_[0])

        for intent, intent_data in self.bot_config["intents"].items():
            for example in intent_data["examples"]:
                intent_names.append(intent)
                intent_examples.append(self.phrase_simplify(example))

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))

        if self._classifier is None:
            self._classifier = LinearSVC(dual=True)

        logging.info("Learning")
        self._classifier.fit(self._vectorizer.fit_transform(intent_examples), intent_names)
        logging.info("Learning finished!")

    def intent_predict(self, request: str) -> str or None:
        """Intent guess"""

        if self._vectorizer is None or self._classifier is None:
            return

        request = self.phrase_simplify(request)
        logging.info(f"Simplified: {request}")

        intent = self._classifier.predict(self._vectorizer.transform([request]))[0]
        logging.info(f'Intent for "{request}": {intent}')
        return intent

    def generate_answer_dialogues(self, replica) -> str or None:
        """Generating best ans from dialogues.txt"""
        replica = self.phrase_simplify(replica)
        words = set(replica.split())
        mini_dataset = [
            pair for word in words if word in self._dialogues_structured for pair in self._dialogues_structured[word]
        ]

        answers = []
        for question, answer in mini_dataset:
            if abs(len(replica) - len(question)) / len(question) < DIALOGUES_THRESHOLD:
                distance = nltk.edit_distance(replica, question)
                distance_weighted = distance / len(question)
                if distance_weighted < DIALOGUES_THRESHOLD:
                    answers.append([distance_weighted, question, answer])

        return min(answers, key=lambda three: three[0])[2] if answers else None
