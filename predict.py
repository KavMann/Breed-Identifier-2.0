from __future__ import annotations

import argparse
import gc
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights

from config import (
    CLASSES_JSON,
    DEVICE,
    FINAL_MODEL,
    IMAGE_SIZE,
    NUM_CLASSES,
    RESIZE_SIZE,
    USE_TEST_TIME_AUGMENTATION,
)
from model import create_model


MIN_CONFIDENCE = 0.45
DEFAULT_TOP_K = 5

REQUEST_TIMEOUT_SECONDS = 15
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

DISPLAY_NAME_OVERRIDES = {
    "leonberg": "Leonberger",
    "labrador": "Labrador Retriever",
    "pembroke": "Pembroke Welsh Corgi",
    "cardigan": "Cardigan Welsh Corgi",
}


def format_breed_name(class_name: str) -> str:
    """
    Convert an internal dataset class name into a readable breed name.
    """
    if class_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[class_name]

    return class_name.replace("_", " ").title()


def is_url(value: str) -> bool:
    """
    Return True when the supplied value is an HTTP or HTTPS URL.
    """
    parsed = urlparse(value)

    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
    )


def get_prediction_transform() -> transforms.Compose:
    """
    Return the preprocessing pipeline used for validation and inference.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT
    weight_transforms = weights.transforms()

    return transforms.Compose(
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


def load_class_mapping(
    classes_json: Path = CLASSES_JSON,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load class mappings from classes.json.
    """
    if not classes_json.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {classes_json}"
        )

    with classes_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict):
        raise RuntimeError(
            "classes.json must contain a JSON object."
        )

    if len(class_to_idx) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"but classes.json contains {len(class_to_idx)}."
        )

    idx_to_class = {
        int(class_index): class_name
        for class_name, class_index in class_to_idx.items()
    }

    expected_indices = set(range(NUM_CLASSES))
    actual_indices = set(idx_to_class.keys())

    if actual_indices != expected_indices:
        raise RuntimeError(
            "Class indices must be continuous from "
            f"0 to {NUM_CLASSES - 1}."
        )

    return class_to_idx, idx_to_class


def load_predictor(
    model_path: Path = FINAL_MODEL,
) -> tuple[nn.Module, dict[int, str]]:
    """
    Load the model and class mapping.

    The model should be loaded once and reused for repeated predictions.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Deployment model not found: {model_path}\n"
            "Run export_model.py first."
        )

    _, idx_to_class = load_class_mapping()

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "The deployment model file has an invalid format."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "The deployment model does not contain "
            "'model_state_dict'."
        )

    checkpoint_classes = checkpoint.get("num_classes")

    if (
        checkpoint_classes is not None
        and checkpoint_classes != NUM_CLASSES
    ):
        raise RuntimeError(
            f"Model expects {checkpoint_classes} classes, "
            f"but config.py specifies {NUM_CLASSES}."
        )

    model_state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(model_state_dict)

    del model_state_dict
    del checkpoint
    gc.collect()

    model = model.to(DEVICE)
    model.eval()

    warm_up_model(model)

    return model, idx_to_class


def warm_up_model(model: nn.Module) -> None:
    """
    Warm up CUDA and cuDNN so the first measured prediction is faster.
    """
    if DEVICE.type != "cuda":
        return

    dummy_input = torch.zeros(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast(
                device_type="cuda",
                enabled=True,
            ):
                model(dummy_input)

    torch.cuda.synchronize()


def validate_download_size(
    response: requests.Response,
) -> None:
    """
    Validate the remote image size using Content-Length when available.
    """
    content_length = response.headers.get("Content-Length")

    if not content_length:
        return

    try:
        size_bytes = int(content_length)
    except ValueError:
        return

    if size_bytes > MAX_IMAGE_SIZE_BYTES:
        max_size_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)

        raise ValueError(
            f"The remote image is too large. "
            f"Maximum allowed size is {max_size_mb:.0f} MB."
        )


def download_image(
    image_url: str,
) -> Image.Image:
    """
    Download and open an image from a direct HTTP or HTTPS URL.
    """
    try:
        response = requests.get(
            image_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/150.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "image/avif,image/webp,image/apng,"
                    "image/svg+xml,image/*,*/*;q=0.8"
                ),
            },
        )

        response.raise_for_status()
        validate_download_size(response)

        content_type = (
            response.headers
            .get("Content-Type", "")
            .split(";")[0]
            .strip()
            .lower()
        )

        if content_type and content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError(
                "The URL did not return a supported image. "
                f"Returned Content-Type: {content_type}"
            )

        image_data = bytearray()

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):
            if not chunk:
                continue

            image_data.extend(chunk)

            if len(image_data) > MAX_IMAGE_SIZE_BYTES:
                max_size_mb = (
                    MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
                )

                raise ValueError(
                    f"The downloaded image exceeds "
                    f"{max_size_mb:.0f} MB."
                )

        if not image_data:
            raise ValueError(
                "The URL returned an empty response."
            )

        with Image.open(BytesIO(image_data)) as image:
            return image.convert("RGB")

    except requests.Timeout as error:
        raise RuntimeError(
            "The image download timed out."
        ) from error

    except requests.TooManyRedirects as error:
        raise RuntimeError(
            "The image URL caused too many redirects."
        ) from error

    except requests.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        raise RuntimeError(
            f"The image server returned HTTP "
            f"status {status_code}."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not download the image: {error}"
        ) from error

    except UnidentifiedImageError as error:
        raise ValueError(
            "The URL response is not a valid supported image."
        ) from error

    except OSError as error:
        raise RuntimeError(
            f"Unable to decode the remote image: {error}"
        ) from error


def open_local_image(
    image_path: Path,
) -> Image.Image:
    """
    Open a local image safely.
    """
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"The supplied path is not a file: {image_path}"
        )

    file_size = image_path.stat().st_size

    if file_size > MAX_IMAGE_SIZE_BYTES:
        max_size_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)

        raise ValueError(
            f"The image is larger than the "
            f"{max_size_mb:.0f} MB limit."
        )

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    except UnidentifiedImageError as error:
        raise ValueError(
            f"The file is not a valid supported image: "
            f"{image_path}"
        ) from error

    except OSError as error:
        raise RuntimeError(
            f"Unable to open the image: {error}"
        ) from error


def open_image(
    image_source: str | Path,
) -> Image.Image:
    """
    Open an image from either a local path or direct image URL.
    """
    source_text = str(image_source).strip()

    if not source_text:
        raise ValueError(
            "The image source cannot be empty."
        )

    if is_url(source_text):
        return download_image(source_text)

    return open_local_image(
        Path(source_text)
    )


@torch.inference_mode()
def predict_pil_image(
    image: Image.Image,
    model: nn.Module,
    idx_to_class: dict[int, str],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Predict dog breeds from an already-loaded PIL image.
    """
    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    top_k = min(top_k, NUM_CLASSES)

    transform = get_prediction_transform()

    normalized_image = image.convert("RGB")
    input_tensors = [transform(normalized_image)]

    if USE_TEST_TIME_AUGMENTATION:
        input_tensors.append(
            transform(ImageOps.mirror(normalized_image))
        )

    input_tensor = torch.stack(input_tensors).to(
        DEVICE,
        non_blocking=True,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.autocast(
        device_type=DEVICE.type,
        enabled=DEVICE.type == "cuda",
    ):
        logits = model(input_tensor).mean(
            dim=0,
            keepdim=True,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    top_probabilities, top_indices = probabilities.topk(
        top_k,
        dim=1,
    )

    predictions: list[dict[str, Any]] = []

    for probability, class_index in zip(
        top_probabilities[0],
        top_indices[0],
    ):
        index = int(class_index.item())
        confidence = float(probability.item())
        raw_name = idx_to_class[index]

        predictions.append(
            {
                "class_index": index,
                "class_name": raw_name,
                "display_name": format_breed_name(
                    raw_name
                ),
                "confidence": confidence,
                "confidence_percent": round(
                    confidence * 100,
                    2,
                ),
            }
        )

    best_prediction = predictions[0]

    return {
        "best_prediction": best_prediction,
        "predictions": predictions,
        "low_confidence": (
            best_prediction["confidence"]
            < MIN_CONFIDENCE
        ),
        "minimum_confidence": MIN_CONFIDENCE,
        "inference_time_ms": round(
            inference_time_ms,
            2,
        ),
        "device": str(DEVICE),
    }


def predict_image(
    image_source: str | Path,
    model: nn.Module,
    idx_to_class: dict[int, str],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Predict dog breeds from either a local path or image URL.
    """
    image = open_image(image_source)

    return predict_pil_image(
        image=image,
        model=model,
        idx_to_class=idx_to_class,
        top_k=top_k,
    )


def print_predictions(
    result: dict[str, Any],
) -> None:
    """
    Display prediction results in the terminal.
    """
    print()
    print("Predictions")
    print("-" * 62)

    for rank, prediction in enumerate(
        result["predictions"],
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{prediction['display_name']:<38}"
            f"{prediction['confidence_percent']:>7.2f}%"
        )

    print("-" * 62)

    best_prediction = result["best_prediction"]

    print(
        f"Best prediction: "
        f"{best_prediction['display_name']}"
    )
    print(
        f"Confidence: "
        f"{best_prediction['confidence_percent']:.2f}%"
    )
    print(
        f"Inference time: "
        f"{result['inference_time_ms']:.2f} ms"
    )
    print(
        f"Device: {result['device']}"
    )

    if result["low_confidence"]:
        print()
        print(
            "Warning: Low-confidence prediction."
        )
        print(
            "The image may contain a mixed breed, "
            "an unsupported breed, or no dog."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict dog breeds from a local image "
            "or direct image URL."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help=(
            "Local image path or direct HTTP/HTTPS "
            "image URL."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of predictions to display. "
            f"Default: {DEFAULT_TOP_K}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("=" * 62)
    print("DOG BREED PREDICTION")
    print("=" * 62)
    print(f"Loading model from: {FINAL_MODEL}")
    print(f"Using device: {DEVICE}")
    print(f"Image source: {args.image}")

    model, idx_to_class = load_predictor()

    result = predict_image(
        image_source=args.image,
        model=model,
        idx_to_class=idx_to_class,
        top_k=args.top_k,
    )

    print_predictions(result)


if __name__ == "__main__":
    main()
