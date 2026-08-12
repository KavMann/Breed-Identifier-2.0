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

## Render Deployment

This repo includes `render.yaml`.

Render settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn wsgi:app --workers 1 --timeout 180`
- Environment variable: `DOGBREED_DEVICE=cpu`
- Optional environment variable: `GEMINI_API_KEY`

The Gemini key should be added in Render's environment settings, not committed to Git.
