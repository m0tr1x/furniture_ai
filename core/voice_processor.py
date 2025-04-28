import logging
import os
import json
from pathlib import Path
import subprocess

from vosk import Model, KaldiRecognizer


class VoiceProcessor:
    def __init__(self, model_path: str):
        self.sample_rate = 16000
        self.logger = logging.getLogger(__name__)
        model_path = Path(model_path)

        # Проверка структуры модели
        required = {
            'am/final.mdl': 'Основная акустическая модель',
            'graph/HCLr.fst': 'Граф декодирования',
            'ivector/final.ie': 'i-vector extractor'
        }

        for file, desc in required.items():
            if not (model_path / file).exists():
                error = f"{desc} не найдена: {model_path / file}"
                self.logger.error(error)
                raise FileNotFoundError(error)

        try:
            self.model = Model(str(model_path))
            self.logger.info(f"Модель Vosk загружена из {model_path}")
        except Exception as e:
            self.logger.critical(f"Ошибка загрузки модели: {str(e)}")
            raise

    def convert_to_wav(self, input_path: str) -> str:
        """Конвертирует входной файл в WAV формат с нужными параметрами"""
        output_path = "temp.wav"
        try:
            result = subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",  # <- добавлено!
                "-y",
                output_path
            ], check=True, capture_output=True)

            self.logger.info(f"Конвертация прошла успешно: {input_path} -> {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка ffmpeg: {e.stderr.decode()}")
            raise RuntimeError(f"Ошибка конвертации аудио: {e.stderr.decode()}")

    def recognize(self, audio_path: str) -> str:
        """Распознает речь из аудиофайла"""
        temp_wav = None

        try:
            if not audio_path.endswith('.wav'):
                temp_wav = self.convert_to_wav(audio_path)
                path_to_use = temp_wav
            else:
                path_to_use = audio_path

            recognizer = KaldiRecognizer(self.model, self.sample_rate)

            text_chunks = []

            with open(path_to_use, "rb") as f:
                while True:
                    data = f.read(4000)
                    if not data:
                        break
                    if recognizer.AcceptWaveform(data):
                        res = json.loads(recognizer.Result())
                        if res.get("text"):
                            text_chunks.append(res["text"])

            # Финальный результат
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get("text"):
                text_chunks.append(final_result["text"])

            # Собираем полный текст
            recognized_text = " ".join(text_chunks).strip()
            return recognized_text if recognized_text else "Не удалось распознать речь"

        except Exception as e:
            self.logger.error(f"Ошибка распознавания: {str(e)}")
            return f"Ошибка: {str(e)}"

        finally:
            # Чистим только временные файлы
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
