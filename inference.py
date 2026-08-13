from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch
from dotenv import load_dotenv
from flask import Request
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

from config import (
    DEVICE,
    DOG_REJECTION_ENABLED,
    DOG_REJECTION_THRESHOLD,
    GRADCAM_ENABLED,
    IMAGE_SIZE,
    RESIZE_SIZE,
)
from predict import (
    ALLOWED_CONTENT_TYPES,
    DEFAULT_TOP_K,
    MAX_IMAGE_SIZE_BYTES,
    MIN_CONFIDENCE,
    download_image,
    is_url,
    load_predictor,
    predict_pil_image,
)


load_dotenv()

CONFIDENCE_LABELS = {
    "high": 0.75,
    "moderate": MIN_CONFIDENCE,
}


@dataclass(frozen=True)
class BreedInformation:
    description: str
    temperament: str
    care_requirements: str


@dataclass(frozen=True)
class DogDetectionResult:
    enabled: bool
    available: bool
    is_dog: bool
    dog_probability: float | None
    dog_probability_percent: float | None
    top_imagenet_label: str | None
    reason: str


def classify_confidence(confidence: float) -> str:
    if confidence >= CONFIDENCE_LABELS["high"]:
        return "High"

    if confidence >= CONFIDENCE_LABELS["moderate"]:
        return "Moderate"

    return "Low"


def _validate_content_type(content_type: str | None) -> None:
    normalized = (content_type or "").split(";")[0].strip().lower()

    if normalized not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            "Please provide a JPEG, PNG, WebP, BMP, or TIFF image."
        )


def open_uploaded_image(file: FileStorage) -> Image.Image:
    if not file or not file.filename:
        raise ValueError("Please choose an image file.")

    _validate_content_type(file.mimetype)

    data = file.read(MAX_IMAGE_SIZE_BYTES + 1)
    file.stream.seek(0)

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(f"Uploaded image must be {max_mb:.0f} MB or smaller.")

    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a valid image.") from error
    except OSError as error:
        raise ValueError(f"Unable to decode uploaded image: {error}") from error


def normalize_prediction(result: dict[str, Any]) -> dict[str, Any]:
    best = result["best_prediction"]

    return {
        "breed": best["display_name"],
        "class_name": best["class_name"],
        "confidence": best["confidence"],
        "confidence_percent": best["confidence_percent"],
        "confidence_label": classify_confidence(best["confidence"]),
        "top_five": result["predictions"],
        "inference_time_ms": result["inference_time_ms"],
        "low_confidence": result["low_confidence"],
        "minimum_confidence_percent": round(MIN_CONFIDENCE * 100, 2),
        "device": result["device"],
    }


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class DogDetector:
    DOG_CLASS_INDICES = set(range(151, 269))

    def __init__(self) -> None:
        self.enabled = DOG_REJECTION_ENABLED
        self.threshold = DOG_REJECTION_THRESHOLD
        self.available = False
        self.model = None
        self.transform = None
        self.categories: list[str] = []
        self.error: str | None = None

        if not self.enabled:
            return

        try:
            from torchvision.models import (
                EfficientNet_V2_S_Weights,
                efficientnet_v2_s,
            )

            weights = EfficientNet_V2_S_Weights.DEFAULT
            self.model = efficientnet_v2_s(weights=weights).to(DEVICE)
            self.model.eval()
            self.transform = weights.transforms()
            self.categories = list(weights.meta.get("categories", []))
            self.available = True
        except Exception as error:
            self.error = str(error)

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> DogDetectionResult:
        if not self.enabled:
            return DogDetectionResult(
                enabled=False,
                available=False,
                is_dog=True,
                dog_probability=None,
                dog_probability_percent=None,
                top_imagenet_label=None,
                reason="Dog rejection is disabled.",
            )

        if not self.available or self.model is None or self.transform is None:
            return DogDetectionResult(
                enabled=True,
                available=False,
                is_dog=True,
                dog_probability=None,
                dog_probability_percent=None,
                top_imagenet_label=None,
                reason=(
                    "Dog detector is unavailable, so the image was allowed "
                    "through to breed classification."
                ),
            )

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
        ):
            logits = self.model(tensor)

        probabilities = torch.softmax(logits, dim=1)[0]
        dog_indices = torch.tensor(
            sorted(self.DOG_CLASS_INDICES),
            device=probabilities.device,
        )
        dog_probability = float(probabilities[dog_indices].sum().item())
        top_probability, top_index = probabilities.max(dim=0)
        top_index_int = int(top_index.item())
        top_label = (
            self.categories[top_index_int]
            if top_index_int < len(self.categories)
            else str(top_index_int)
        )
        top_is_dog = top_index_int in self.DOG_CLASS_INDICES
        is_dog = dog_probability >= self.threshold or top_is_dog

        if is_dog:
            reason = "ImageNet dog gate accepted the image."
        else:
            reason = (
                "The image does not look enough like a dog for reliable "
                "breed classification."
            )

        return DogDetectionResult(
            enabled=True,
            available=True,
            is_dog=is_dog,
            dog_probability=dog_probability,
            dog_probability_percent=round(dog_probability * 100, 2),
            top_imagenet_label=top_label,
            reason=reason,
        )


class GradCamGenerator:
    def __init__(self, model) -> None:
        from torchvision import transforms
        from torchvision.models import EfficientNet_V2_S_Weights

        self.model = model
        self.target_layer = model.features[-1]
        weights = EfficientNet_V2_S_Weights.DEFAULT
        weight_transforms = weights.transforms()
        self.model_transform = transforms.Compose(
            [
                transforms.Resize(RESIZE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=weight_transforms.mean,
                    std=weight_transforms.std,
                ),
            ]
        )
        self.display_transform = transforms.Compose(
            [
                transforms.Resize(RESIZE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
            ]
        )

    def generate(
        self,
        image: Image.Image,
        class_index: int,
    ) -> dict[str, Any]:
        if not GRADCAM_ENABLED:
            return {
                "enabled": False,
                "available": False,
                "image": None,
                "reason": "Grad-CAM is disabled.",
            }

        activations = []
        gradients = []

        def forward_hook(_module, _inputs, output):
            activations.append(output.detach())

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

        try:
            input_tensor = (
                self.model_transform(image.convert("RGB"))
                .unsqueeze(0)
                .to(DEVICE)
            )

            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=False,
                ):
                    logits = self.model(input_tensor)
                    score = logits[0, class_index]
                score.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture tensors.")

            activation = activations[-1]
            gradient = gradients[-1]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam[0, 0]
            cam_min = cam.min()
            cam_max = cam.max()

            if float((cam_max - cam_min).item()) <= 1e-8:
                raise RuntimeError("Grad-CAM heatmap was blank.")

            cam = (cam - cam_min) / (cam_max - cam_min)
            overlay = create_heatmap_overlay(
                base_image=self.display_transform(image.convert("RGB")),
                heatmap=cam.detach().cpu().numpy(),
            )

            return {
                "enabled": True,
                "available": True,
                "image": image_to_data_uri(overlay),
                "reason": "Grad-CAM generated from the final EfficientNet feature layer.",
            }
        except Exception as error:
            return {
                "enabled": True,
                "available": False,
                "image": None,
                "reason": f"Grad-CAM unavailable: {error}",
            }
        finally:
            forward_handle.remove()
            backward_handle.remove()
            self.model.zero_grad(set_to_none=True)


def create_heatmap_overlay(
    *,
    base_image: Image.Image,
    heatmap: Any,
) -> Image.Image:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import colormaps

    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap = colormaps.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(colored).resize(
        base_image.size,
        Image.Resampling.BILINEAR,
    )
    base_image = base_image.convert("RGB")
    return Image.blend(base_image, heatmap_image, alpha=0.42)


class GeminiBreedInfoClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.enabled_by_config = (
            os.getenv("GEMINI_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self._client = None
        self._cache: dict[str, BreedInformation] = {}

        if not self.api_key or not self.enabled_by_config:
            return

        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        except Exception:
            self._client = None

    @property
    def enabled(self) -> bool:
        return self.enabled_by_config and self._client is not None

    def get_breed_information(self, breed: str) -> BreedInformation:
        cache_key = breed.lower().strip()

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.enabled:
            return fallback_breed_information(
                breed,
                (
                    "Gemini breed information is disabled or no API key is "
                    "configured."
                ),
            )

        prompt = f"""
Return concise dog breed information for {breed}.
Respond as strict JSON with exactly these string keys:
description, temperament, care_requirements.
Do not include Markdown or code fences.
"""

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            information = _parse_breed_information(response.text)
            self._cache[cache_key] = information
            return information
        except Exception as error:
            return fallback_breed_information(
                breed,
                summarize_gemini_error(error),
            )


def summarize_gemini_error(error: Exception) -> str:
    error_text = str(error)

    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
        return (
            "Gemini quota is exhausted for the configured API key. "
            "The classifier still works; breed notes are using a local fallback."
        )

    if "API_KEY" in error_text or "permission" in error_text.lower():
        return (
            "Gemini could not be reached with the configured API key. "
            "Breed notes are using a local fallback."
        )

    return (
        "Gemini breed information is temporarily unavailable. "
        "Breed notes are using a local fallback."
    )


def fallback_breed_information(
    breed: str,
    reason: str,
) -> BreedInformation:
    return BreedInformation(
        description=(
            f"{breed} was the closest match from the image classifier. "
            "Use this as a model prediction rather than a veterinary or "
            "pedigree confirmation."
        ),
        temperament=(
            "Temperament varies by individual dog, training, socialization, "
            "age, and health. Meet the dog and ask a breeder, shelter, or "
            "veterinarian for breed-specific context."
        ),
        care_requirements=(
            f"{reason} General care still applies: provide regular exercise, "
            "grooming, training, preventive veterinary care, and a diet suited "
            "to the dog's age and size."
        ),
    )


def _parse_breed_information(text: str) -> BreedInformation:
    import json

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return BreedInformation(
            description=cleaned,
            temperament="No structured temperament data was returned.",
            care_requirements="No structured care data was returned.",
        )

    return BreedInformation(
        description=str(payload.get("description", "")).strip(),
        temperament=str(payload.get("temperament", "")).strip(),
        care_requirements=str(payload.get("care_requirements", "")).strip(),
    )


class InferenceService:
    def __init__(self) -> None:
        self.model, self.idx_to_class = load_predictor()
        self.dog_detector = DogDetector()
        self.gradcam = (
            GradCamGenerator(self.model)
            if GRADCAM_ENABLED
            else None
        )
        self.gemini = GeminiBreedInfoClient()

    def predict_uploaded_file(self, file: FileStorage) -> dict[str, Any]:
        image = open_uploaded_image(file)
        return self._predict_image(image)

    def predict_url(self, image_url: str) -> dict[str, Any]:
        image_url = image_url.strip()

        if not image_url:
            raise ValueError("Please enter a direct image URL.")

        if not is_url(image_url):
            raise ValueError("Please enter a valid HTTP or HTTPS image URL.")

        image = download_image(image_url)
        return self._predict_image(image)

    def _predict_image(self, image: Image.Image) -> dict[str, Any]:
        dog_detection = self.dog_detector.detect(image)

        if dog_detection.enabled and not dog_detection.is_dog:
            return {
                "is_dog": False,
                "breed": "Not a dog",
                "class_name": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "confidence_label": "Low",
                "top_five": [],
                "inference_time_ms": 0.0,
                "low_confidence": True,
                "minimum_confidence_percent": round(
                    MIN_CONFIDENCE * 100,
                    2,
                ),
                "device": str(DEVICE),
                "dog_detection": dog_detection.__dict__,
                "gradcam": {
                    "enabled": GRADCAM_ENABLED,
                    "available": False,
                    "image": None,
                    "reason": "Grad-CAM skipped because the image was rejected as non-dog.",
                },
                "breed_info": fallback_breed_information(
                    "the submitted image",
                    dog_detection.reason,
                ).__dict__,
            }

        result = predict_pil_image(
            image=image,
            model=self.model,
            idx_to_class=self.idx_to_class,
            top_k=DEFAULT_TOP_K,
        )
        normalized = normalize_prediction(result)
        normalized["is_dog"] = True
        normalized["dog_detection"] = dog_detection.__dict__
        if self.gradcam is None:
            normalized["gradcam"] = {
                "enabled": False,
                "available": False,
                "image": None,
                "reason": "Grad-CAM is disabled.",
            }
        else:
            normalized["gradcam"] = self.gradcam.generate(
                image=image,
                class_index=result["best_prediction"]["class_index"],
            )
        normalized["breed_info"] = self.gemini.get_breed_information(
            normalized["breed"]
        ).__dict__
        return normalized


def validate_request_size(request: Request) -> None:
    content_length = request.content_length

    if content_length and content_length > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(f"Request body must be {max_mb:.0f} MB or smaller.")
