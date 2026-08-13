from __future__ import annotations

import os
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

os.environ.setdefault("DOGBREED_DEVICE", "cpu")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("DOGBREED_TTA", "false")
os.environ.setdefault("DOGBREED_DOG_REJECTION", "true")
os.environ.setdefault("DOGBREED_GRADCAM", "true")

from inference import InferenceService
from predict import MAX_IMAGE_SIZE_BYTES


st.set_page_config(
    page_title="Dog Breed Identifier",
    page_icon="🐾",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading model...")
def get_inference_service() -> InferenceService:
    return InferenceService()


def open_uploaded_image(uploaded_file) -> Image.Image:
    data = uploaded_file.getvalue()

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(f"Uploaded image must be {max_mb:.0f} MB or smaller.")

    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a valid image.") from error


def show_prediction(result: dict) -> None:
    if not result.get("is_dog", True):
        st.error(result["dog_detection"]["reason"])
        return

    label = result["confidence_label"]
    if label == "High":
        st.success(f"{label} confidence")
    elif label == "Moderate":
        st.warning(f"{label} confidence")
    else:
        st.error(f"{label} confidence")

    st.subheader(result["breed"])
    st.caption(
        f"{result['confidence_percent']:.2f}% confidence · "
        f"{result['inference_time_ms']:.2f} ms · {result['device']}"
    )

    if result["low_confidence"]:
        st.warning(
            "Low-confidence prediction. The image may contain a mixed breed, "
            "an unsupported breed, or no clear dog."
        )

    st.markdown("### Top five predictions")
    for prediction in result["top_five"]:
        st.write(
            f"**{prediction['display_name']}** "
            f"{prediction['confidence_percent']:.2f}%"
        )
        st.progress(
            min(max(prediction["confidence"], 0.0), 1.0),
            text=None,
        )

    gradcam = result.get("gradcam", {})
    if gradcam.get("available") and gradcam.get("image"):
        st.markdown("### Grad-CAM visualization")
        st.image(gradcam["image"], use_container_width=True)

    breed_info = result.get("breed_info", {})
    if breed_info:
        st.markdown("### Breed information")
        st.markdown(f"**Description:** {breed_info.get('description', '')}")
        st.markdown(f"**Temperament:** {breed_info.get('temperament', '')}")
        st.markdown(
            f"**Care requirements:** "
            f"{breed_info.get('care_requirements', '')}"
        )


st.title("Dog Breed Identifier")
st.caption("Upload a dog image or paste a direct image URL.")

tab_upload, tab_url = st.tabs(["Upload", "Image URL"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
    )

    if uploaded_file is not None:
        try:
            uploaded_image = open_uploaded_image(uploaded_file)
            st.image(uploaded_image, use_container_width=True)

            if st.button("Predict uploaded image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = get_inference_service()._predict_image(
                        uploaded_image
                    )
                show_prediction(result)
        except Exception as error:
            st.error(str(error))

with tab_url:
    image_url = st.text_input(
        "Direct image URL",
        placeholder="https://example.com/dog.jpg",
    )

    if image_url:
        st.image(image_url, use_container_width=True)

    if st.button("Predict image URL", type="primary"):
        try:
            with st.spinner("Fetching and analyzing image..."):
                result = get_inference_service().predict_url(image_url)
            show_prediction(result)
        except Exception as error:
            st.error(str(error))
