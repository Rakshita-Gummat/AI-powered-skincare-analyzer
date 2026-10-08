# Skincare analyzer
<div align="center">

# AI-Powered Skincare Analyzer

**Upload a face photo. Get a skin condition prediction and a tailored skincare routine.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

##  Overview

The Skincare Analyzer is an end-to-end deep learning application that classifies facial skin conditions from an image and recommends condition-specific skincare treatments. Three architectures (a custom CNN, DenseNet121 and EfficientNetB0) were trained and evaluated on the same dataset, and the best-suited model is served through a Flask web interface.

**Detected conditions**

| Acne | Dark Spots | Normal Skin | Puffy Eyes | Wrinkles |
|:---:|:---:|:---:|:---:|:---:|

##  Features

-  **Image upload** through a clean web UI (PNG, JPG, JPEG, GIF)
-  **Multi-class classification** across 5 skin conditions
-  **Three trained models** compared: CNN, DenseNet121, EfficientNetB0
-  **Personalized treatment suggestions** for every detected condition
-  **Full evaluation suite**: accuracy, precision, recall, F1, confusion matrix, classification report
-  **Safe uploads**: extension validation, `secure_filename`, and user-facing error messages
-  **Persistent models** saved as `.keras` files and loaded once at startup

##  Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, NumPy, Pillow |
| Backend | Flask, Werkzeug |
| Frontend | HTML, CSS, Jinja2 |
| Evaluation & Analysis | scikit-learn, Pandas, Matplotlib, Seaborn |

##  How It Works

```mermaid
flowchart LR
    A[Upload image] --> B[Validate file type]
    B --> C[Resize to 128×128<br/>Normalize to 0–1]
    C --> D[Model inference]
    D --> E[Predicted condition]
    E --> F[Treatment suggestions]
    F --> G[Results page]
```

1. The user uploads a face image on the `/upload` page.
2. Flask validates the file extension and stores it securely in `static/uploads/`.
3. OpenCV reads the image, resizes it to **128×128** and scales pixel values to `[0, 1]`.
4. The model outputs class probabilities, and `argmax` selects the predicted condition.
5. The condition is mapped to a skincare routine (cleansers, active ingredients, sunscreen, lifestyle tips) and displayed alongside the uploaded image.

##  Model Architectures

### 1. Custom CNN: `cnn_model.keras`
- 3 convolutional blocks with ReLU activation and max pooling
- Fully connected dense layers
- Dropout for regularization

### 2. DenseNet121: `densenet_model.keras`
- ImageNet-pretrained DenseNet121 used as a frozen feature extractor
- Global average pooling followed by fully connected layers
- **Used by the web app for inference**

### 3. EfficientNetB0: `efficientnet_model.keras`
- ImageNet-pretrained EfficientNetB0 backbone
- Custom classification head for the 5 skin condition classes

### Training Configuration

| Parameter | Value |
|---|---|
| Input size | 128 × 128 |
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |

### Evaluation

Each model is evaluated on a held-out validation set using accuracy, precision, recall, F1 score, a confusion matrix and a per-class classification report.

##  Dataset

Labeled facial skin images organized by class:

```
DATASET/
├── acne/
├── dark spots/
├── normal skin/
├── puffy eyes/
└── wrinkles/
```

All images are resized to 128×128 and normalized to the `[0, 1]` range before training and inference.

##  Project Structure

```
AI-powered-skincare-analyzer/
├── app.py                      # Flask app: routes, inference, treatment mapping
├── predict_skin_condition.py   # Prediction and treatment helper functions
├── models/                     # Trained .keras models
├── Dataset/DATASET/            # Training images by class
├── static/                     # CSS and uploaded images
├── templates/                  # index.html, upload.html, results.html
├── requirements.txt            # Pinned dependencies
├── LICENSE
└── README.md
```

##  Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rakshita-Gummat/AI-powered-skincare-analyzer.git
cd AI-powered-skincare-analyzer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Ensure `cnn_model.keras`, `densenet_model.keras` and `efficientnet_model.keras` are present in the `models/` directory.

### Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, go to the upload page and submit a face image.

### Use the prediction API directly

```python
from predict_skin_condition import predict_skin_condition, suggest_treatment

condition = predict_skin_condition("densenet", "path/to/image.jpg")
print(condition)
print(suggest_treatment(condition))
```

`model_type` accepts `"cnn"`, `"densenet"` or `"efficientnet"`.

##  Sample Recommendations

| Condition | Example suggestions |
|---|---|
| Acne | Salicylic acid cleanser, benzoyl peroxide, non-comedogenic moisturizer, SPF 30+ |
| Dark spots | Glycolic acid exfoliation, Vitamin C serum, broad-spectrum sunscreen |
| Puffy eyes | Cold compress, caffeine eye cream, reduced salt intake, elevated sleep |
| Wrinkles | Retinol serum at night, peptide moisturizer, daily sunscreen |
| Normal skin | Gentle hydrating cleanser, hyaluronic acid moisturizer, SPF 30+ |

##  Limitations

- Predicts one dominant condition per image (no multi-label output)
- Accuracy depends on image quality, lighting and dataset diversity
- No face detection or cropping step before inference

##  Future Improvements

- [ ] Real-time webcam capture
- [ ] Confidence scores and top-k predictions
- [ ] Automatic face detection and cropping
- [ ] Grad-CAM visual explanations
- [ ] Additional skin conditions and a larger, more diverse dataset
- [ ] Cloud deployment (Docker, Render, Hugging Face Spaces)

##  Disclaimer

This project is for **educational and informational purposes only**. It is not a medical device and does not replace professional dermatological advice.

## Contributing

Contributions are welcome. Fork the repo, create a feature branch and open a pull request.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.


---

<div align="center"> If you found this project useful, consider giving it a star.</div>
