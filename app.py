
"""
Flask API simple para inferencia con Oxford 102 Flowers
-----------------------------------------------------
- Endpoint POST /predict
- Carga un modelo .pt (state_dict)
- Devuelve nombre real de la flor + probabilidad

Estructura esperada:
.
├── app.py                  (este archivo)
├── best_model.pt           (checkpoint)
├── cat_to_name.json        (mapping Oxford)
"""

import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS

# ------------------ CONFIG ------------------
MODEL_PATH = "best_model.pt"
CAT_TO_NAME_PATH = "cat_to_name.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONE = "efficientnet_b0"  # debe coincidir con el entrenamiento
# --------------------------------------------


# ------------------ TRANSFORMS ------------------
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
# -----------------------------------------------


# ------------------ MODEL ------------------
def build_model(backbone: str, num_classes: int = 102):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )
    else:
        raise ValueError("Backbone no soportado")

    return model


def load_model():
    model = build_model(BACKBONE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model
# --------------------------------------------


# ------------------ LOAD ASSETS ------------------
with open(CAT_TO_NAME_PATH, "r") as f:
    cat_to_name = json.load(f)

model = load_model()
# ------------------------------------------------


# ------------------ FLASK APP ------------------
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera un archivo enviado como form-data con key 'image'
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió archivo 'image'"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Archivo no es una imagen válida"}), 400

    x = val_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs))
    prob = float(probs[top_idx])

    oxford_id = str(top_idx + 1)
    flower_name = cat_to_name.get(oxford_id, "unknown")

    return jsonify({
        "prediction": flower_name,
        "probability": prob,
        "class_id": int(oxford_id)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
