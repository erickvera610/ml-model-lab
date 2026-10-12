FROM python:3.12-slim
WORKDIR /app
COPY requirements-benchmark.txt .
RUN pip install --no-cache-dir -r requirements-benchmark.txt
COPY . .
RUN useradd --create-home appuser
USER appuser
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" || exit 1
CMD ["python", "app.py"]
