FROM python:3.12-slim

WORKDIR /app

# Evita cachés y mejora logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Código
COPY . .

EXPOSE 8000

CMD ["python", "scripts/start_api.py"]
