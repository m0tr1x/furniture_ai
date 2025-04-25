FROM python:3.12-slim

# 1. Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libatomic1 \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория
WORKDIR /app

# 3. Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копирование модели (если не используете volume)
RUN mkdir -p /app/model && \
    wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip -O /tmp/model.zip && \
    unzip /tmp/model.zip -d /tmp && \
    mv /tmp/vosk-model-small-ru-0.22/* /app/model/ && \
    rm -rf /tmp/model.zip /tmp/vosk-model-small-ru-0.22

# 5. Копирование исходного кода
COPY . .

# 6. Запуск приложения
CMD ["python", "main.py"]