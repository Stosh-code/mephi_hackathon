FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY run.py /app/run.py
ENTRYPOINT ["python", "-m", "doe.cli"]
CMD ["--help"]
