FROM python:3.12-slim

# 1. Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libatomic1 \
    g++ \
    make \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория
WORKDIR /app

# 3. Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копирование исходного кода
COPY . .

# 5. Запуск приложения
CMD ["python", "main.py"]
