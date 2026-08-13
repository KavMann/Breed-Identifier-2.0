FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DOGBREED_DEVICE=cpu \
    TORCH_NUM_THREADS=2 \
    DOGBREED_TTA=false \
    DOGBREED_DOG_REJECTION=true \
    DOGBREED_GRADCAM=true \
    GEMINI_MODEL=gemini-1.5-flash

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements-hf.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-hf.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "240"]
