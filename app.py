from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import pandas as pd
import os
from google import genai
import logging
import json
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configure the API key
client = genai.Client(api_key="AIzaSyC9")

app = Flask(__name__)

# Load the pre-trained dog breed classification model
model_classification = load_model("Final_dog_identification.h5")

# Load class_indices mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}


csv_file_path = os.path.join(os.path.dirname(__file__), "labels.csv")
colnames = ['Id', 'breed']
df_labels = pd.read_csv(csv_file_path, names=colnames)

# Define breed list globally
breed_dict = list(df_labels['breed'].value_counts().keys())
new_list = sorted(breed_dict, reverse=True)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_breed = None
    breed_info = None
    breed_needs = None
    image_filename = None
    image_path = None

    if request.method == "POST":
        image_url = request.form.get("image_url", "").strip()

        # If file was uploaded
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            image_filename = "uploaded_image.jpg"
            image_path = os.path.join("static", image_filename)
            file.save(image_path)

        # If image URL is provided instead
        elif image_url:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image_filename = "uploaded_image.jpg"
                    image_path = os.path.join("static", image_filename)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                else:
                    return render_template("index.html", predicted_breed="Failed to fetch image from URL.")
            except Exception as e:
                return render_template("index.html", predicted_breed=f"Error: {e}")

        # If no input
        else:
            return render_template("index.html", predicted_breed="Please upload or paste an image.")

        # Predict
        predicted_breed, confidence = predict_breed(image_path)
        if predicted_breed is None:
            predicted_breed = "Could not identify breed. Please check the image."
        else:
            # Call your Gemini breed_info + breed_needs functions here
            breed_info = get_breed_characteristics(predicted_breed)
            breed_needs = get_breed_needs(predicted_breed)

    return render_template("index.html",
                           predicted_breed=predicted_breed,
                           breed_info=breed_info,
                           breed_needs=breed_needs,
                           image_path=image_filename)

def predict_breed(image_path, confidence_threshold=0.5):
    img_array = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), (299, 299))
    img_array = preprocess_input(np.expand_dims(np.array(img_array[...,::-1].astype(np.float32)).copy(), axis=0))
    
    # Perform prediction
    pred_val = model_classification.predict(np.array(img_array, dtype="float32"))

    # Check if pred_val is valid (not None or empty)
    if pred_val is None or len(pred_val) == 0:
        logging.error(f"Prediction failed for image: {image_path}")
        return None, None  # Returning None as the prediction is invalid

    # Get the max probability (confidence) and the predicted breed
    max_confidence = np.max(pred_val)
    predicted_index = int(np.argmax(pred_val))
    predicted_breed = idx_to_class[predicted_index]


    # Log the confidence
    logging.info(f"Predicted Breed: {predicted_breed}, Confidence: {max_confidence * 100:.2f}%")

    # Check if confidence is below the threshold
    if max_confidence < confidence_threshold:
        logging.warning(f"Prediction confidence too low for breed {predicted_breed}: {max_confidence * 100:.2f}%")
        return None, max_confidence  # Confidence too low to identify breed

    return predicted_breed, max_confidence

def get_breed_characteristics(breed):
    prompt = f"""
    Provide a concise summary of the basic characteristics of the {breed} dog breed.
    Respond in clean and well-structured HTML and no need to give titles. Use:
    - <h4> for headings
    - <ul> and <li> for lists
    - <strong> for emphasis
    Avoid using Markdown. Only return pure HTML.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return clean_html_response(response.text).strip()
    except Exception as e:
        print(f"Error getting breed characteristics: {e}")
        return "Could not retrieve breed characteristics at this time."


def get_breed_needs(breed):
    prompt2 = f"""
    What are the essential care and lifestyle needs of a {breed} dog?
    Include exercise, grooming, training, health concerns, etc.
    No need to include section titles(eg {breed} needs).
    Return the response as structured HTML only. Use:
    - <h4> for headings
    - <ul><li> for lists
    - <strong> for highlighting
    Avoid Markdown or plain text formatting.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt2
        )
        return clean_html_response(response.text).strip()
    except Exception as e:
        print(f"Error getting breed needs: {e}")
        return "Could not retrieve breed needs at this time."

def clean_html_response(text):
    if text.startswith("```html"):
        text = text.replace("```html", "").strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

@app.route('/static/<path:path>')
def static_css(path):
    return app.send_static_file(os.path.join('static', path))

if __name__ == "__main__":
    app.run(debug=True)
