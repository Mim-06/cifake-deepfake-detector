from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

_model = None


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    model = get_model()
    if model is None:
        raise FileNotFoundError(
            "Model not found. Train the model first with: python train_model.py"
        )

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    # EfficientNetB0 includes its input preprocessing in the model.
    probability = float(model.predict(array, verbose=0)[0][0])

    # Training uses REAL=0 and FAKE=1.
    label = "FAKE" if probability >= 0.5 else "REAL"
    confidence = probability if label == "FAKE" else 1.0 - probability

    return label, confidence * 100.0, probability


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please choose an image."
        elif not allowed_file(file.filename):
            error = "Please upload PNG, JPG, JPEG, or WEBP."
        else:
            filename = secure_filename(file.filename)
            save_path = UPLOAD_DIR / filename
            file.save(save_path)
            image_url = "/static/uploads/" + filename

            try:
                label, confidence, probability = predict_image(save_path)
                result = {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "probability": round(probability, 4),
                }
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=image_url,
        model_exists=MODEL_PATH.exists(),
    )


if __name__ == "__main__":
    print("CIFAKE web application starting...")
    print(f"Model path: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print("WARNING: model is not trained yet. Run: python train_model.py")
    app.run(host="127.0.0.1", port=5000, debug=True)
