from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import re
import json
from pathlib import Path
import logging


class NLProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_phrases()
        self._init_pipeline()
        self._prepare_training_data()

    def _load_phrases(self):
        """Загрузка и валидация данных"""
        try:
            config_path = Path(__file__).parent.parent / 'data' / 'phrases.json'
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "intents" not in data:
                raise ValueError("Key 'intents' not found in phrases.json")

            self.intents = data["intents"]
            self.logger.info(f"Loaded {len(self.intents)} intents")

            # Валидация данных
            for intent, content in self.intents.items():
                if not isinstance(content, dict) or 'examples' not in content:
                    raise ValueError(f"Intent '{intent}' has invalid structure")
                if len(content['examples']) < 2:
                    self.logger.warning(
                        f"Intent '{intent}' has only {len(content['examples'])} examples (minimum 2 recommended)")

        except Exception as e:
            self.logger.error(f"Error loading phrases: {e}")
            raise

    def _init_pipeline(self):
        """Безопасная инициализация пайплайна"""
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 3),  # Более безопасный диапазон
                lowercase=True,
                max_features=2000)),  # Уменьшенное количество фич

            ('classifier', LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=10000))  # Увеличенное количество итераций
        ])
        self.logger.info("Pipeline initialized with safe parameters")

    def _prepare_training_data(self):
        """Обучение с дополнительными проверками"""
        try:
            X = []
            y = []

            for intent, data in self.intents.items():
                examples = data['examples']
                X.extend(examples)
                y.extend([intent] * len(examples))

            # Финализируем проверки
            if len(set(y)) < 2:
                raise ValueError(f"Need at least 2 classes, got {len(set(y))}")
            if len(X) < 5:
                raise ValueError(f"Need at least 5 examples, got {len(X)}")

            self.pipeline.fit(X, y)
            self.logger.info(f"Model trained on {len(X)} examples ({len(set(y))} classes)")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Очистка текста"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

    def predict(self, text: str) -> str:
        """Предсказание с защитой от ошибок"""
        try:
            if not hasattr(self.pipeline, 'steps'):
                raise RuntimeError("Model not trained")

            processed = self._preprocess_text(text)
            intent = self.pipeline.predict([processed])[0]
            self.logger.debug(f"Predicted '{intent}' for '{text}'")
            return intent

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "error"