import logging
import os
import json
from pathlib import Path

from vosk import Model, KaldiRecognizer
import subprocess


class VoiceProcessor:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        model_path = Path(model_path)

        # Подробная проверка модели
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
            from vosk import Model
            self.model = Model(str(model_path))  # Явное преобразование в str
            self.logger.info(f"Модель загружена из {model_path}")
        except Exception as e:
            self.logger.critical(f"Ошибка загрузки: {str(e)}")
            raise

    def convert_to_wav(self, input_path):
        """Конвертирует аудио в WAV формат"""
        output_path = "temp.wav"
        try:
            subprocess.run([
                "ffmpeg",
                "-i", input_path,
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-y",
                output_path
            ], check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ошибка конвертации: {e.stderr.decode()}")

    def recognize(self, audio_path):
        """Распознает речь из аудиофайла"""
        try:
            # Конвертируем в WAV если нужно
            if not audio_path.endswith('.wav'):
                audio_path = self.convert_to_wav(audio_path)

            recognizer = KaldiRecognizer(self.model, self.sample_rate)

            with open(audio_path, "rb") as f:
                while True:
                    data = f.read(4000)
                    if not data:
                        break
                    if recognizer.AcceptWaveform(data):
                        pass

            result = json.loads(recognizer.FinalResult())
            return result.get("text", "Не удалось распознать речь")

        except Exception as e:
            return f"Ошибка: {str(e)}"
        finally:
            if 'audio_path' in locals() and audio_path != "temp.wav" and os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")