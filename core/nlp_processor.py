import numpy as np
from sklearn.base import BaseEstimator
from config import NLP_CONFIG


class NLProcessor(BaseEstimator):
    def __init__(self):
        self.vectorizer = self._init_vectorizer()
        self.classifier = self._init_classifier()

    def _init_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=0.001
        )

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.vectorizer.transform(X), y)
        return self

    def predict(self, text):
        return self.classifier.predict(self.vectorizer.transform([text]))[0]