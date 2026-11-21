FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY ml/ ml/
COPY data/ data/
# Train the model while building, so the image is self-contained
RUN python ml/train.py
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5000/health')" || exit 1
CMD ["gunicorn", "--chdir", "app", "--bind", "0.0.0.0:5000", "app:app"]
