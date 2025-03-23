import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
app.secret_key = "your_secret_key"  # Replace with your secret key

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the pre-trained models once at startup
# Make sure these files exist in the 'models' folder
cnn_model = load_model(os.path.join("models", "cnn_model.keras"))
densenet_model = load_model(os.path.join("models", "densenet_model.keras"))
efficientnet_model = load_model(os.path.join("models", "efficientnet_model.keras"))
# Categories of skin conditions
categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Function to predict the skin condition
def predict_skin_condition(model_type, img_path):
    # Select the model
    if model_type == 'cnn':
        model = cnn_model
    elif model_type == 'densenet':
        model = densenet_model
    elif model_type == 'efficientnet':
        model = efficientnet_model
    else:
        raise ValueError("Invalid model type. Choose 'cnn', 'densenet', or 'efficientnet'.")

    # Preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0

    # Perform prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return categories[class_index]

# Treatment suggestions
def suggest_treatment(condition):
    treatments = {
        'acne': [
            "Use a salicylic acid-based cleanser twice a day.",
            "Apply a benzoyl peroxide cream to affected areas.",
            "Moisturize with a non-comedogenic, oil-free lotion.",
            "Always wear sunscreen (SPF 30+) during the day."
        ],
        'dark spots': [
            "Use a gentle exfoliating cleanser with glycolic acid.",
            "Apply a Vitamin C serum daily to brighten skin tone.",
            "Protect your skin with a broad-spectrum sunscreen.",
            "Consider professional treatments like chemical peels."
        ],
        'normal skin': [
            "Cleanse with a gentle hydrating cleanser twice a day.",
            "Use a lightweight moisturizer with hyaluronic acid.",
            "Apply sunscreen with at least SPF 30 in the morning.",
            "Maintain a healthy diet and stay hydrated."
        ],
        'puffy eyes': [
            "Apply a cold compress for a few minutes in the morning.",
            "Use an under-eye cream with caffeine or retinol.",
            "Avoid excess salt and stay hydrated throughout the day.",
            "Sleep with your head slightly elevated."
        ],
        'wrinkles': [
            "Cleanse with a mild, hydrating cleanser.",
            "Apply a retinol-based serum at night.",
            "Use a peptide-rich moisturizer to boost skin elasticity.",
            "Always wear sunscreen to prevent further aging."
        ],
    }
    return treatments.get(condition, ["Consult a dermatologist for personalized advice."])

# Landing page (index.html)
@app.route('/')
def index():
    return render_template("index.html")

# Upload page (upload.html)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part in the request.")
            print("DEBUG: No file part in request")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == "":
            flash("No file selected.")
            print("DEBUG: No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            print(f"DEBUG: File saved at {file_path}")

            try:
                condition = predict_skin_condition('densenet', file_path)
                treatment = suggest_treatment(condition)
                print(f"DEBUG: Predicted condition - {condition}")
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                print(f"DEBUG: Error processing image - {str(e)}")
                return redirect(request.url)

            return render_template("results.html",
                                   image_path=url_for('static', filename=f"uploads/{filename}"),
                                   condition=condition,
                                   treatment=treatment)
        else:
            flash("Invalid file type. Please upload an image file.")
            print("DEBUG: Invalid file type")
            return redirect(request.url)

    return render_template("upload.html")

if __name__ == '__main__':
    # Turn on debugging for development
    app.run(debug=True)
