FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

CMD exec uvicorn model_app:app port=$PORT --host=0.0.0.0