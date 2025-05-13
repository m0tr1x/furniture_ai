docker build -t my-furniture-bot:latest .
Start for Macos
docker run -v "${PWD}/models/vosk-model-small-ru-0.22:/app/models" my-furniture-bot:latest
Start for win
docker run -v "${PWD}\models\vosk-model-small-ru-0.22:/app/models" my-furniture-bot:latest