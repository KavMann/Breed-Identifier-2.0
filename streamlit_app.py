from __future__ import annotations

import base64
import html
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


BACKGROUND_IMAGE_URL = (
    "https://www.thesprucepets.com/thmb/nAfZqzn_BhMmJ2z2rmbGkO439xM=/"
    "4000x0/filters:no_upscale():strip_icc()/"
    "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg"
)


st.set_page_config(
    page_title="Dog Breed Identifier",
    page_icon=":dog:",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            min-height: 100%;
        }}

        .stApp {{
            color: #291313;
            background-color: #201a17;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            z-index: -2;
            inset: -28px;
            background:
                linear-gradient(135deg, rgba(34, 22, 18, 0.76), rgba(120, 64, 33, 0.46), rgba(46, 93, 84, 0.42)),
                url("{BACKGROUND_IMAGE_URL}") center center / cover no-repeat;
            filter: blur(12px) saturate(0.95) brightness(0.78);
            transform: scale(1.04);
            opacity: 0.95;
            pointer-events: none;
        }}

        .stApp::after {{
            content: "";
            position: fixed;
            z-index: -1;
            inset: 0;
            background:
                linear-gradient(180deg, rgba(255, 246, 238, 0.14), rgba(32, 26, 23, 0.58)),
                linear-gradient(120deg, transparent 0%, rgba(255, 160, 122, 0.18) 34%, rgba(46, 93, 84, 0.2) 68%, transparent 100%);
            background-size: 100% 100%, 220% 220%;
            animation: backgroundDrift 16s ease-in-out infinite alternate;
            pointer-events: none;
        }}

        @keyframes backgroundDrift {{
            from {{ background-position: center, 0% 50%; }}
            to {{ background-position: center, 100% 50%; }}
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .block-container {{
            max-width: 760px;
            padding-top: 2.7rem;
            padding-bottom: 3.2rem;
        }}

        .main-card,
        .result-card,
        .details-card {{
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.62), rgba(255, 245, 236, 0.34)),
                rgba(255, 255, 255, 0.24);
            backdrop-filter: blur(34px) saturate(1.18);
            border: 1px solid rgba(255, 255, 255, 0.72);
            box-shadow:
                0 34px 90px rgba(21, 15, 13, 0.42),
                inset 0 1px 0 rgba(255, 255, 255, 0.68),
                inset 0 -1px 0 rgba(255, 255, 255, 0.18);
            border-radius: 14px;
            overflow: hidden;
            color: #291313;
        }}

        .main-card {{
            border-radius: 14px 14px 0 0;
        }}

        .main-title {{
            margin: 0;
            padding: 26px 16px 20px;
            text-align: center;
            font-size: clamp(1.55rem, 5vw, 2.25rem);
            font-weight: 900;
            letter-spacing: 0;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.56), rgba(255, 245, 236, 0.22));
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
            border-bottom: 1px solid rgba(255, 255, 255, 0.36);
        }}

        .main-body {{
            padding: 22px 20px 18px;
        }}

        .st-key-input_shell {{
            margin-top: -1px;
            padding: 0 20px 22px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.62), rgba(255, 245, 236, 0.34)),
                rgba(255, 255, 255, 0.24);
            backdrop-filter: blur(34px) saturate(1.18);
            border-left: 1px solid rgba(255, 255, 255, 0.72);
            border-right: 1px solid rgba(255, 255, 255, 0.72);
            border-bottom: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 0 0 14px 14px;
            box-shadow:
                0 34px 90px rgba(21, 15, 13, 0.42),
                inset 0 -1px 0 rgba(255, 255, 255, 0.18);
            color: #291313;
        }}

        .st-key-input_shell [data-testid="stHorizontalBlock"] {{
            align-items: center;
        }}

        .upload-box {{
            border: 2px dashed rgba(117, 91, 75, 0.35);
            padding: 24px;
            text-align: center;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.58), rgba(255, 240, 228, 0.36));
            backdrop-filter: blur(12px);
            box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.5),
                0 12px 28px rgba(43, 26, 20, 0.08);
            margin-bottom: 14px;
        }}

        .upload-title {{
            margin-bottom: 5px;
            color: #2e201d;
            font-size: 1.03rem;
            font-weight: 800;
        }}

        .upload-note {{
            margin-bottom: 0;
            color: #785b4b;
            font-size: 0.88rem;
            line-height: 1.45;
        }}

        .input-row-label {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            color: #785b4b;
            font-weight: 900;
        }}

        [data-testid="stFileUploader"] {{
            width: 100%;
        }}

        [data-testid="stFileUploader"] section {{
            padding: 0;
            border: 0;
            background: transparent;
        }}

        [data-testid="stFileUploader"] section > div {{
            padding: 0;
        }}

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] svg,
        [data-testid="stFileUploaderDropzoneInstructions"] {{
            display: none;
        }}

        [data-testid="stFileUploader"] button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            width: 100%;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(117, 91, 75, 0.22);
            color: #4a3028;
            font-weight: 800;
            box-shadow: 0 8px 18px rgba(43, 26, 20, 0.1);
        }}

        [data-testid="stFileUploader"] button::after {{
            content: "Choose File";
        }}

        [data-testid="stFileUploader"] button p {{
            display: none;
        }}

        .stTextInput input {{
            min-height: 46px;
            border: 1px solid rgba(117, 91, 75, 0.22);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #291313;
        }}

        div.stButton > button {{
            background: #2f5d54;
            color: #fff;
            min-height: 46px;
            padding: 11px 28px;
            border: 1px solid rgba(47, 93, 84, 0.2);
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(47, 93, 84, 0.22);
        }}

        div.stButton {{
            display: flex;
            justify-content: center;
        }}

        div.stButton > button:hover {{
            background: #274f48;
            color: #fff;
            border: 1px solid rgba(47, 93, 84, 0.2);
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(47, 93, 84, 0.26);
        }}

        [data-testid="stImage"] img {{
            margin: 12px auto;
            border-radius: 12px;
            max-height: 390px;
            object-fit: contain;
            box-shadow: 0 20px 46px rgba(43, 26, 20, 0.22);
            background: rgba(255, 255, 255, 0.48);
            border: 1px solid rgba(255, 255, 255, 0.65);
        }}

        .message-card {{
            margin-top: 15px;
            padding: 10px 12px;
            border-radius: 8px;
            background: rgba(230, 246, 239, 0.82);
            color: #225d4f;
            border: 1px solid rgba(34, 93, 79, 0.1);
            font-weight: 700;
            text-align: center;
        }}

        .result-card {{
            margin-top: 16px;
            padding: 18px;
            text-align: center;
        }}

        .details-card {{
            margin-top: 22px;
            padding: 20px;
            text-align: left;
        }}

        .confidence-label {{
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 800;
            background: rgba(139, 69, 19, 0.16);
            color: #8b4513;
            margin-bottom: 8px;
        }}

        .confidence-label.high {{
            background: rgba(37, 113, 82, 0.16);
            color: #226647;
        }}

        .confidence-label.moderate {{
            background: rgba(160, 111, 31, 0.16);
            color: #875914;
        }}

        .confidence-label.low {{
            background: rgba(150, 37, 37, 0.14);
            color: #8f2424;
        }}

        .result-card h2 {{
            font-size: 1.15rem;
            margin: 8px 0 4px;
        }}

        .result-card h3 {{
            font-size: 1.8rem;
            margin: 0 0 8px;
            overflow-wrap: anywhere;
            color: #291313;
        }}

        .summary-metrics {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }}

        .summary-metrics span {{
            min-width: 132px;
            padding: 8px 12px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.58);
            box-shadow: inset 0 0 0 1px rgba(139, 69, 19, 0.08);
            text-align: center;
        }}

        .summary-metrics strong {{
            display: block;
            line-height: 1.15;
        }}

        .summary-metrics small {{
            display: block;
            margin-top: 2px;
            color: #785b4b;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }}

        .low-confidence {{
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            color: #8b4513;
            background: rgba(255, 224, 204, 0.82);
            border-left: 4px solid rgba(139, 69, 19, 0.55);
            font-weight: 700;
            text-align: left;
        }}

        .dog-gate-message {{
            margin: 14px auto 0;
            padding: 10px 12px;
            border-radius: 8px;
            font-weight: 700;
            line-height: 1.45;
            text-align: center;
            max-width: 650px;
            }}

        .dog-gate-message.accepted {{
            color: #226647;
            background: rgba(229, 248, 236, 0.82);
            border-left: 4px solid rgba(34, 102, 71, 0.72);
        }}

        .dog-gate-message.rejected {{
            color: #8f2424;
            background: rgba(255, 235, 230, 0.86);
            border-left: 4px solid rgba(143, 36, 36, 0.72);
        }}

        .details-card h3 {{
            margin-top: 16px;
            margin-bottom: 8px;
            font-size: 1.05rem;
            font-weight: 800;
            color: #2e201d;
        }}

        .details-card h3:first-child {{
            margin-top: 0;
        }}

        .prediction-bars {{
            display: grid;
            gap: 12px;
            margin-bottom: 20px;
        }}

        .prediction-row {{
            padding: 11px 12px;
            border-radius: 8px;
            background: rgba(255, 250, 246, 0.72);
            box-shadow: inset 0 0 0 1px rgba(139, 69, 19, 0.06);
        }}

        .prediction-topline {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 5px;
            color: #291313;
        }}

        .bar-track {{
            height: 12px;
            border-radius: 999px;
            background: rgba(139, 69, 19, 0.16);
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(to right, #7a4021, #d8794e, #2e5d54);
            box-shadow: 0 0 12px rgba(216, 121, 78, 0.28);
        }}

        .gradcam-section {{
            margin: 22px 0 24px;
            padding: 14px;
            border-radius: 10px;
            background: rgba(255, 245, 236, 0.56);
            box-shadow: inset 0 0 0 1px rgba(139, 69, 19, 0.07), 0 14px 34px rgba(43, 26, 20, 0.1);
            text-align: center;
        }}

        .gradcam-section p {{
            text-align: center;
            color: #785b4b;
            font-size: 0.92rem;
            margin-bottom: 0;
        }}

        .gradcam-section img {{
            width: 100%;
            max-width: 460px;
            display: block;
            margin: 10px auto 12px;
            border-radius: 12px;
            box-shadow: 0 18px 42px rgba(43, 26, 20, 0.22);
            border: 1px solid rgba(255, 255, 255, 0.55);
        }}

        .breed-info-cards {{
            display: grid;
            gap: 12px;
            margin-top: 20px;
        }}

        .breed-info-card {{
            padding: 14px 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.44);
            border: 1px solid rgba(255, 255, 255, 0.48);
            box-shadow:
                0 10px 24px rgba(43, 26, 20, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.42);
            backdrop-filter: blur(12px);
        }}

        .breed-info-card h3 {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 0;
            margin-bottom: 7px;
        }}

        .breed-info-card h3::before {{
            content: "";
            width: 8px;
            height: 8px;
            flex: 0 0 8px;
            border-radius: 999px;
            background: linear-gradient(135deg, #d8794e, #2e5d54);
        }}

        .breed-info-card p {{
            color: #4a3028;
            line-height: 1.55;
            margin-bottom: 0;
        }}

        @media (max-width: 760px) {{
            .block-container {{
                padding: 18px 10px 2.5rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model...")
def get_inference_service() -> InferenceService:
    return InferenceService()


def escape(value: object) -> str:
    return html.escape(str(value or ""))


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


def data_uri_to_image(data_uri: str) -> Image.Image:
    if "," not in data_uri:
        raise ValueError("Invalid Grad-CAM image data.")

    _, encoded = data_uri.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def confidence_class(label: str) -> str:
    if label == "High":
        return "high"
    if label == "Moderate":
        return "moderate"
    return "low"


def render_prediction_bars(predictions: list[dict]) -> str:
    rows = []

    for prediction in predictions:
        percent = float(prediction["confidence_percent"])
        rows.append(
            (
                '<div class="prediction-row">'
                '<div class="prediction-topline">'
                f'<span>{escape(prediction["display_name"])}</span>'
                f"<strong>{percent:.2f}%</strong>"
                "</div>"
                '<div class="bar-track">'
                '<div class="bar-fill" '
                f'style="width: {min(max(percent, 0.0), 100.0):.2f}%">'
                "</div>"
                "</div>"
                "</div>"
            )
        )

    return "\n".join(rows)


def render_dog_gate(result: dict) -> str:
    detection = result.get("dog_detection")

    if not detection or not detection.get("enabled"):
        return ""

    probability = detection.get("dog_probability_percent")
    probability_text = (
        "unavailable"
        if probability is None
        else f"{float(probability):.2f}%"
    )

    if result.get("is_dog", True):
        return (
            '<div class="dog-gate-message accepted">'
            f"Input validation passed. Dog likelihood: {probability_text}."
            "</div>"
        )

    return (
        '<div class="dog-gate-message rejected">'
        f"Input validation rejected this image. {escape(detection.get('reason'))}"
        "</div>"
    )


def show_prediction(result: dict) -> None:
    if not result.get("is_dog", True):
        st.markdown(
            (
                '<div class="result-card">'
                '<span class="confidence-label low">Rejected</span>'
                "<h3>Not a dog</h3>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    label = result["confidence_label"]
    low_warning = (
        (
            '<div class="low-confidence">'
            "Low-confidence prediction. The image may contain a mixed breed, "
            "an unsupported breed, or no dog."
            "</div>"
        )
        if result["low_confidence"]
        else ""
    )

    st.markdown(
        (
            '<section class="result-card">'
            f'<span class="confidence-label {confidence_class(label)}">'
            f"{escape(label)}</span>"
            "<h2>Breed for this Dog is:</h2>"
            f'<h3>{escape(result["breed"])}</h3>'
            '<div class="summary-metrics">'
            "<span>"
            f'<strong>{float(result["confidence_percent"]):.2f}%</strong>'
            "<small>confidence</small>"
            "</span>"
            "<span>"
            f'<strong>{float(result["inference_time_ms"]):.2f} ms</strong>'
            "<small>analysis time</small>"
            "</span>"
            "</div>"
            f"{low_warning}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )

    gradcam = result.get("gradcam", {})
    breed_info = result.get("breed_info", {})

    st.markdown(
        (
            '<section class="details-card">'
            "<h3>Top Five Predictions:</h3>"
            '<div class="prediction-bars">'
            f'{render_prediction_bars(result.get("top_five", []))}'
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )

    if gradcam.get("available") and gradcam.get("image"):
        st.markdown(
            (
                '<section class="details-card">'
                '<div class="gradcam-section">'
                "<h3>Model Attention Map:</h3>"
                "</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )
        try:
            st.image(
                data_uri_to_image(gradcam["image"]),
                use_container_width=True,
            )
        except Exception:
            st.image(gradcam["image"], use_container_width=True)
        st.markdown(
            (
                '<section class="details-card" style="margin-top: 0;">'
                '<div class="gradcam-section" style="margin-top: 0;">'
                "<p>Highlighted regions indicate the image areas that most "
                "influenced the breed prediction.</p>"
                "</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )
    elif gradcam.get("enabled"):
        st.markdown(
            (
                '<section class="details-card">'
                '<div class="gradcam-section">'
                "<h3>Model Attention Map:</h3>"
                "<p>"
                f'{escape(gradcam.get("reason", "Grad-CAM is unavailable."))}'
                "</p>"
                "</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        (
            '<section class="details-card">'
            '<div class="breed-info-cards">'
            '<article class="breed-info-card">'
            "<h3>Breed Description</h3>"
            "<p>"
            f'{escape(breed_info.get("description", "No description available."))}'
            "</p>"
            "</article>"
            '<article class="breed-info-card">'
            "<h3>Temperament</h3>"
            "<p>"
            f'{escape(breed_info.get("temperament", "No temperament information available."))}'
            "</p>"
            "</article>"
            '<article class="breed-info-card">'
            "<h3>Care Requirements</h3>"
            "<p>"
            f'{escape(breed_info.get("care_requirements", "No care information available."))}'
            "</p>"
            "</article>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def show_input_validation(result: dict) -> None:
    validation_message = render_dog_gate(result)

    if not validation_message:
        return

    st.markdown(validation_message, unsafe_allow_html=True)


def show_main_card() -> None:
    st.markdown(
        """
        <section class="main-card">
            <h1 class="main-title">DOG BREED IDENTIFICATION</h1>
            <div class="main-body">
                <div class="upload-box">
                    <p class="upload-title">Drag an image into this area</p>
                    <p class="upload-note">Use the controls below to browse locally or paste a direct image URL.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_input_controls() -> None:
    with st.container(key="input_shell"):
        file_col, divider_col, url_col = st.columns(
            [1.2, 0.3, 4.2],
            gap="small",
        )

        with file_col:
            uploaded_file = st.file_uploader(
                "Choose File",
                type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                help="JPEG, PNG, WebP, BMP, or TIFF up to 15 MB.",
                label_visibility="collapsed",
            )

        with divider_col:
            st.markdown(
                '<div class="input-row-label">or</div>',
                unsafe_allow_html=True,
            )

        with url_col:
            image_url = st.text_input(
                "Direct image URL",
                placeholder="Paste a direct image URL",
                label_visibility="collapsed",
            )

        uploaded_image = None

        if uploaded_file is not None:
            uploaded_image = open_uploaded_image(uploaded_file)
            st.image(uploaded_image, use_container_width=True)
        elif image_url:
            st.image(image_url, use_container_width=True)

        if st.button("Predict Breed", type="primary", key="predict"):
            if uploaded_image is None and not image_url.strip():
                st.error(
                    "Please choose an image file or paste a direct image URL."
                )
                return

            with st.spinner(
                "Checking the image and preparing the breed analysis..."
            ):
                if uploaded_image is not None:
                    st.session_state["prediction_result"] = (
                        get_inference_service()._predict_image(uploaded_image)
                    )
                else:
                    st.session_state["prediction_result"] = (
                        get_inference_service().predict_url(image_url.strip())
                    )


inject_styles()
show_main_card()

try:
    show_input_controls()
except Exception as error:
    st.error(str(error))

result = st.session_state.get("prediction_result")

if result:
    st.markdown(
        '<div class="message-card">Analysis complete.</div>',
        unsafe_allow_html=True,
    )
    show_input_validation(result)
    show_prediction(result)
