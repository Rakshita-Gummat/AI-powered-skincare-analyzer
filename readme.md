# AI-Powered Skincare Analyzer

A deep learning web app that classifies facial skin conditions from an uploaded image and returns tailored skincare suggestions.

**Detected classes:** acne · dark spots · normal skin · puffy eyes · wrinkles

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)
![Flask](https://img.shields.io/badge/Flask-3.1.0-black)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- TODO: add demo GIF / screenshots of the landing, upload and results pages -->

---

## Features

- Upload a face image (PNG, JPG, JPEG, GIF) through a web UI
- Multi-class skin condition prediction using a trained CNN
- Three models trained and compared: custom CNN, DenseNet, EfficientNet
- Condition-specific skincare recommendations on the results page
- Input validation, secure filename handling, and error feedback via Flask flash messages

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras (CNN, DenseNet, EfficientNet) |
| Image processing | OpenCV, NumPy, Pillow |
| Backend | Flask, Werkzeug |
| Frontend | HTML, CSS (Jinja2 templates) |
| Analysis / evaluation | Pandas, scikit-learn, Matplotlib, Seaborn |

## How It Works

1. User uploads an image on `/upload`.
2. Flask validates the extension and saves the file securely to `static/uploads/`.
3. The image is read with OpenCV, resized to **128×128**, and normalised to [0, 1].
4. The selected model outputs class probabilities; `argmax` gives the predicted condition. The web app currently uses **DenseNet**.
5. The predicted condition is mapped to a set of skincare suggestions and shown on the results page with the uploaded image.

```
Upload → Validate → Preprocess (128×128, /255) → Model inference → Class + Treatment tips → Results page
```

## Models

| Model | File | Notes |
|---|---|---|
| Custom CNN | `models/cnn_model.keras` | Baseline |
| DenseNet | `models/densenet_model.keras` | Used by the web app |
| EfficientNet | `models/efficientnet_model.keras` | Available in prediction script |

### Results

<!-- TODO: fill in from your training notebook -->
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN | – | – | – | – |
| DenseNet | – | – | – | – |
| EfficientNet | – | – | – | – |

## Dataset

- Five classes: acne, dark spots, normal skin, puffy eyes, wrinkles
- Located in `Dataset/DATASET`
- <!-- TODO: source, total images, per-class counts, train/val/test split, augmentation -->

## Project Structure

```
AI-powered-skincare-analyzer/
├── app.py                      # Flask app (routes, inference, treatment mapping)
├── predict_skin_condition.py   # Standalone prediction + treatment helpers
├── cnn_model.keras             # Trained CNN
├── densenet_model.keras        # Trained DenseNet
├── models/                     # Model weights loaded by the app
├── Dataset/DATASET/            # Training data
├── static/                     # CSS, uploads
├── templates/                  # index.html, upload.html, results.html
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Rakshita-Gummat/AI-powered-skincare-analyzer.git
cd AI-powered-skincare-analyzer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Make sure `cnn_model.keras`, `densenet_model.keras` and `efficientnet_model.keras` are inside the `models/` folder.

## Usage

```bash
python app.py
```

Open `http://127.0.0.1:5000`, go to the upload page, and submit a face image.

**Standalone prediction:**

```python
from predict_skin_condition import predict_skin_condition, suggest_treatment

condition = predict_skin_condition('densenet', 'path/to/image.jpg')
print(condition, suggest_treatment(condition))
```

`model_type` accepts `'cnn'`, `'densenet'` or `'efficientnet'`.

## Limitations

- Predicts one dominant condition per image; no multi-label output
- Accuracy depends on image quality, lighting and dataset diversity
- No face detection or cropping step before inference
- <!-- TODO: add any known dataset bias / skin-tone coverage notes -->

## Disclaimer

This tool is for educational and informational purposes only. It is not a medical device and does not replace advice from a dermatologist.

## Future Improvements

- Model selection / ensemble from the UI
- Confidence scores and top-k predictions
- Face detection and cropping (e.g., OpenCV Haar / MediaPipe)
- Grad-CAM explainability
- Deployment (Docker, Render / HF Spaces)

## License

MIT, see [LICENSE](LICENSE).

## Author

**Rakshita Gummat** · [GitHub](https://github.com/Rakshita-Gummat)
