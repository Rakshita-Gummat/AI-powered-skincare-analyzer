# AI-Powered Skincare Analyzer

A deep learning web app that analyzes facial skin conditions and recommends condition-specific skincare treatments. Three image classification models (custom CNN, DenseNet121, EfficientNetB0) were trained and compared; the Flask app serves predictions through an upload-and-results UI.

**Detected classes:** acne · dark spots · normal skin · puffy eyes · wrinkles

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)
![Flask](https://img.shields.io/badge/Flask-3.1.0-black)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- TODO: add screenshots / demo GIF of landing, upload and results pages -->

---

## Features

- Upload a face image (PNG, JPG, JPEG, GIF) through a Flask web interface
- Multi-class skin condition prediction
- Three trained models: CNN, DenseNet121, EfficientNetB0
- Personalized treatment suggestions per detected condition
- Evaluation with accuracy, precision, recall, F1, confusion matrix and classification report
- Trained models saved as `.keras` files and reloaded at app startup
- Secure filename handling, file-type validation, and flash-message error feedback

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Image processing | OpenCV, NumPy, Pillow |
| Backend | Flask, Werkzeug |
| Frontend | HTML, CSS (Jinja2 templates) |
| Evaluation | scikit-learn, Matplotlib, Seaborn, Pandas |

## Dataset

Labeled facial skin images across five classes:
DATASET/
├── acne/
├── dark spots/
├── normal skin/
├── puffy eyes/
└── wrinkles/

- Preprocessing: resize to **128×128**, normalize pixel values to [0, 1]
- Evaluated on a separate validation set
- !-- TODO: data source, total image count, per-class counts, split ratio, augmentation -->

## Model Architectures

**1. Custom CNN** (cnn_model.keras)
- 3 convolutional layers with ReLU and max pooling
- Fully connected dense layers with dropout for regularization

**2. DenseNet121** (densenet_model.keras`)
- ImageNet-pretrained DenseNet121 as a frozen feature extractor
- Global average pooling + fully connected classification head
- Used by the web app for inference

**3. EfficientNetB0** (efficientnet_model.keras`)
- ImageNet-pretrained EfficientNetB0 backbone
- Custom top layers for the 5-class output

### Training Configuration

| Parameter | Value |
|---|---|
| Image size | 128×128 |
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Categorical crossentropy |

## How It Works

1. User uploads an image at `/upload`.
2. Flask validates the extension and saves it securely to `static/uploads/`.
3. OpenCV preprocesses the image; the model returns class probabilities and `argmax` selects the condition.
4. The condition is mapped to a skincare routine (cleansers, actives, sunscreen, lifestyle tips) and rendered with the image on the results page.

## Installation

```bash
git clone https://github.com/Rakshita-Gummat/AI-powered-skincare-analyzer.git
cd AI-powered-skincare-analyzer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place cnn_model.keras, densenet_model.keras and  efficientnet_model.keras in the models/ folder.

## Usage

**Web app**
```bash
python app.py
```
Open `http://127.0.0.1:5000` and upload a face image.

**Python API**
```python
from predict_skin_condition import predict_skin_condition, suggest_treatment

condition = predict_skin_condition('densenet', 'path/to/image.jpg')
print(condition, suggest_treatment(condition))
```
`model_type` accepts `'cnn'`, `'densenet'` or `'efficientnet'`.

## Training (optional)

Set `data_dir` in each script to your dataset path, then run:

```bash
python cnn_model.py
python densenet_model.py
python efficientnet_model.py
```

## Project Structure
├── app.py # Flask app: routes, inference, treatment mapping
├── predict_skin_condition.py # Prediction + treatment helper functions
├── cnn_model.py # CNN training and evaluation
├── densenet_model.py # DenseNet121 training and evaluation
├── efficientnet_model.py # EfficientNetB0 training and evaluation
├── models/ # Saved .keras models
├── Dataset/DATASET/ # Training data
├── static/ # CSS and uploads
├── templates/ # index.html, upload.html, results.html
├── requirements.txt
├── LICENSE
└── README.md

## Disclaimer

For educational and informational purposes only. Not a medical device; does not replace advice from a dermatologist.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Rakshita Gummat** · [GitHub](https://github.com/Rakshita-Gummat)
