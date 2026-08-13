from __future__ import annotations

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


st.set_page_config(
    page_title="Dog Breed Identifier",
    page_icon="DB",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #151110;
            --panel: rgba(40, 31, 27, 0.74);
            --panel-strong: rgba(52, 40, 35, 0.86);
            --line: rgba(255, 236, 218, 0.12);
            --text: #fff7f0;
            --muted: #cdbbb1;
            --accent: #df7648;
            --accent-2: #f2b56f;
            --good: #58d487;
            --warn: #f4c060;
            --bad: #ff756b;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(223, 118, 72, 0.22), transparent 28rem),
                radial-gradient(circle at 82% 8%, rgba(242, 181, 111, 0.12), transparent 24rem),
                linear-gradient(135deg, #151110 0%, #1d1715 48%, #100d0c 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .app-hero {
            padding: 1.4rem 0 1.1rem;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2.4rem, 5vw, 4.7rem);
            line-height: 0.96;
            font-weight: 850;
            margin: 0 0 1rem;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.06rem;
            max-width: 44rem;
            line-height: 1.65;
        }

        .glass-card {
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(18px);
            border-radius: 18px;
            padding: 1.35rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.35rem;
        }

        .mini-metric {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
        }

        .mini-metric b {
            display: block;
            color: var(--text);
            font-size: 1.05rem;
        }

        .mini-metric span {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .result-card {
            background: linear-gradient(180deg, var(--panel-strong), rgba(30, 23, 21, 0.9));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.45rem;
            margin-top: 1.2rem;
            box-shadow: 0 20px 52px rgba(0, 0, 0, 0.26);
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.36rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.95rem;
        }

        .status-high {
            color: #0f311b;
            background: linear-gradient(135deg, #6ff29d, #baf5cf);
        }

        .status-moderate {
            color: #3a2600;
            background: linear-gradient(135deg, #ffd675, #fff0b7);
        }

        .status-low {
            color: #3c0805;
            background: linear-gradient(135deg, #ff887f, #ffd0cc);
        }

        .breed-title {
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1;
            font-weight: 850;
            margin: 0;
        }

        .result-meta {
            color: var(--muted);
            margin-top: 0.6rem;
            font-size: 0.95rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 850;
            margin: 1.5rem 0 0.85rem;
        }

        .prediction-row {
            display: grid;
            grid-template-columns: minmax(9rem, 1fr) auto;
            gap: 1rem;
            align-items: center;
            margin: 0.85rem 0 0.35rem;
        }

        .prediction-name {
            color: var(--text);
            font-weight: 750;
        }

        .prediction-value {
            color: var(--muted);
            font-weight: 800;
        }

        .bar-track {
            height: 0.58rem;
            width: 100%;
            background: rgba(255, 255, 255, 0.07);
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 0.85rem;
        }

        .info-box {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 1rem;
            min-height: 9rem;
        }

        .info-box b {
            color: var(--accent-2);
            display: block;
            margin-bottom: 0.45rem;
        }

        .info-box p {
            color: var(--muted);
            line-height: 1.55;
            margin: 0;
        }

        div.stButton > button {
            background: linear-gradient(135deg, var(--accent), #b75e3d);
            color: white;
            border: 0;
            border-radius: 12px;
            padding: 0.65rem 1.1rem;
            font-weight: 800;
            box-shadow: 0 12px 30px rgba(223, 118, 72, 0.26);
        }

        div.stButton > button:hover {
            border: 0;
            color: white;
            filter: brightness(1.08);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px dashed rgba(255, 236, 218, 0.25);
            border-radius: 16px;
            padding: 0.8rem;
        }

        [data-testid="stImage"] img {
            border-radius: 16px;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.25);
        }

        @media (max-width: 780px) {
            .block-container {
                padding-top: 2rem;
            }

            .metric-row,
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def esc(value: object) -> str:
    return html.escape(str(value or ""))


def render_prediction_bars(predictions: list[dict]) -> str:
    rows = []

    for prediction in predictions:
        confidence = float(prediction["confidence"])
        width = max(0.0, min(confidence * 100, 100.0))
        rows.append(
            f"""
            <div class="prediction-row">
                <div class="prediction-name">{esc(prediction['display_name'])}</div>
                <div class="prediction-value">{prediction['confidence_percent']:.2f}%</div>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {width:.2f}%"></div>
            </div>
            """
        )

    return "\n".join(rows)


def confidence_class(label: str) -> str:
    if label == "High":
        return "status-high"
    if label == "Moderate":
        return "status-moderate"
    return "status-low"


def show_prediction(result: dict) -> None:
    if not result.get("is_dog", True):
        st.error(result["dog_detection"]["reason"])
        return

    label = result["confidence_label"]
    low_warning = (
        """
        <div style="margin-top: 1rem; color: #ffe0db; background: rgba(255, 117, 107, 0.12);
            border: 1px solid rgba(255, 117, 107, 0.25); border-radius: 13px; padding: 0.9rem 1rem;">
            Low-confidence prediction. The image may contain a mixed breed, an unsupported breed,
            or no clear dog.
        </div>
        """
        if result["low_confidence"]
        else ""
    )

    st.markdown(
        f"""
        <div class="result-card">
            <span class="status-pill {confidence_class(label)}">{esc(label)} confidence</span>
            <h2 class="breed-title">{esc(result['breed'])}</h2>
            <div class="result-meta">
                {result['confidence_percent']:.2f}% confidence /
                {result['inference_time_ms']:.2f} ms /
                {esc(result['device'])}
            </div>
            {low_warning}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="section-title" style="margin-top: 0;">Top five predictions</div>
            {render_prediction_bars(result["top_five"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    gradcam = result.get("gradcam", {})
    if gradcam.get("available") and gradcam.get("image"):
        st.markdown(
            '<div class="section-title">Grad-CAM visualization</div>',
            unsafe_allow_html=True,
        )
        st.image(gradcam["image"], use_container_width=True)

    breed_info = result.get("breed_info", {})
    if breed_info:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-title" style="margin-top: 0;">Breed information</div>
                <div class="info-grid">
                    <div class="info-box">
                        <b>Description</b>
                        <p>{esc(breed_info.get("description", ""))}</p>
                    </div>
                    <div class="info-box">
                        <b>Temperament</b>
                        <p>{esc(breed_info.get("temperament", ""))}</p>
                    </div>
                    <div class="info-box">
                        <b>Care requirements</b>
                        <p>{esc(breed_info.get("care_requirements", ""))}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_shell() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Computer vision demo</div>
            <h1 class="hero-title">Dog Breed Identifier</h1>
            <p class="hero-copy">
                Upload a dog image or paste a direct image URL. The model returns
                the predicted breed, confidence, top-five alternatives, Grad-CAM
                attention, and concise breed notes.
            </p>
            <div class="metric-row">
                <div class="mini-metric"><b>120</b><span>supported breeds</span></div>
                <div class="mini-metric"><b>Grad-CAM</b><span>attention map</span></div>
                <div class="mini-metric"><b>Top-5</b><span>ranked predictions</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
render_shell()

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div class="section-title" style="margin-top: 0;">Input image</div>
            <p style="color: var(--muted); line-height: 1.55; margin: 0;">
                Upload a file or use a direct image URL. Large images are checked
                before inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_upload, tab_url = st.tabs(["Upload", "Image URL"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            help="JPEG, PNG, WebP, BMP, or TIFF up to 15 MB.",
        )

        if uploaded_file is not None:
            try:
                uploaded_image = open_uploaded_image(uploaded_file)
                st.image(uploaded_image, use_container_width=True)

                if st.button("Predict breed", type="primary", key="upload"):
                    with st.spinner("Analyzing breed traits..."):
                        st.session_state["result"] = (
                            get_inference_service()._predict_image(
                                uploaded_image
                            )
                        )
            except Exception as error:
                st.error(str(error))

    with tab_url:
        image_url = st.text_input(
            "Direct image URL",
            placeholder="https://example.com/dog.jpg",
        )

        if image_url:
            st.image(image_url, use_container_width=True)

        if st.button("Predict breed", type="primary", key="url"):
            try:
                with st.spinner("Fetching and analyzing image..."):
                    st.session_state["result"] = (
                        get_inference_service().predict_url(image_url)
                    )
            except Exception as error:
                st.error(str(error))

with right:
    result = st.session_state.get("result")
    if result:
        show_prediction(result)
    else:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title" style="margin-top: 0;">Ready for inference</div>
                <p style="color: var(--muted); line-height: 1.65; margin-bottom: 0;">
                    Add an image on the left and run prediction. Results will appear here with
                    confidence, ranked alternatives, visual attention, and breed information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
