import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model 

# Load the pre-trained models
cnn_model = load_model('models/cnn_model.keras')
densenet_model = load_model('models/densenet_model.keras')
efficientnet_model = load_model('models/efficientnet_model.keras')

# Categories of skin conditions
categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']

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