---
title: Dog Breed Identifier
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# Dog Breed Identifier

Flask web app for dog breed classification using an exported PyTorch EfficientNet V2-S model.

## Features

- Upload an image or paste a direct image URL.
- Returns breed, confidence, top-five predictions, and inference time.
- Rejects likely non-dog images.
- Shows a Grad-CAM attention map for accepted dog images.
- Adds optional Gemini-generated breed description, temperament, and care notes.

## Required Runtime Files

```text
app.py
config.py
inference.py
model.py
predict.py
wsgi.py
requirements.txt
requirements-hf.txt
requirements-streamlit.txt
Dockerfile
render.yaml
runtime.txt
models/
  classes.json
  dog_breed_classifier.pth
static/
  Comp_1.gif
  css/main.css
templates/
  index.html
```

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

## Hugging Face Spaces Deployment

This is the recommended free-tier target for the full demo. Hugging Face lists
CPU Basic Spaces at 2 vCPU and 16 GB RAM, which is much more suitable for
PyTorch, Grad-CAM, and the ImageNet dog gate than Render's free web service.

Create a new Hugging Face Space with:

- SDK: `Docker`
- Hardware: `CPU Basic`
- App port: `7860`

The included `Dockerfile` enables:

- `DOGBREED_DEVICE=cpu`
- `DOGBREED_DOG_REJECTION=true`
- `DOGBREED_GRADCAM=true`
- `DOGBREED_TTA=false`

Add `GEMINI_API_KEY` as a Space secret, not as a committed file.

## Streamlit Community Cloud Deployment

If Hugging Face Docker Spaces is unavailable on your account, Streamlit
Community Cloud is the next free option to try. It has a higher memory limit
than Render's free web service, but it uses `streamlit_app.py` instead of the
Flask frontend.

Deploy settings:

- Repository: this GitHub repo
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: `3.11`
- Requirements file: `requirements-streamlit.txt`

Add these secrets or environment variables:

- `GEMINI_API_KEY`
- `DOGBREED_DEVICE=cpu`
- `TORCH_NUM_THREADS=1`
- `DOGBREED_TTA=false`
- `DOGBREED_DOG_REJECTION=true`
- `DOGBREED_GRADCAM=true`

## Render Deployment

This repo still includes `render.yaml`, but Render's free web service is too
memory constrained for the full PyTorch demo.

Render settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn wsgi:app --workers 1 --timeout 180`
- Environment variable: `DOGBREED_DEVICE=cpu`
- Environment variable: `TORCH_NUM_THREADS=1`
- Environment variable: `DOGBREED_TTA=false`
- Environment variable: `DOGBREED_DOG_REJECTION=false`
- Environment variable: `DOGBREED_GRADCAM=false`
- Optional environment variable: `GEMINI_API_KEY`

The Gemini key should be added in Render's environment settings, not committed to Git.

Render's free tier has a small memory limit. The included `render.yaml` uses
a lean CPU configuration so only the exported breed model is loaded. Grad-CAM,
test-time augmentation, and the ImageNet dog gate can be enabled locally or on
a larger paid instance.
