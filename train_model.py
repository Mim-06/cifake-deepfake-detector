from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

print("TensorFlow version:", tf.__version__)
print("Training directory:", TRAIN_DIR)
print("Test directory:", TEST_DIR)

for folder in [
    TRAIN_DIR / "REAL",
    TRAIN_DIR / "FAKE",
    TEST_DIR / "REAL",
    TEST_DIR / "FAKE",
]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    subset="training",
    seed=SEED,
    class_names=["REAL", "FAKE"],
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    subset="validation",
    seed=SEED,
    class_names=["REAL", "FAKE"],
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=["REAL", "FAKE"],
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Class mapping: REAL=0, FAKE=1")
print("REAL = 0, FAKE = 1")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="augmentation",
)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

model = keras.Model(inputs, outputs, name="CIFAKE_EfficientNetB0")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

checkpoint_path = MODEL_DIR / "best_model.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

best_model = keras.models.load_model(checkpoint_path, compile=False)

best_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

print("\nFinal test evaluation:")
results = best_model.evaluate(test_ds, verbose=1, return_dict=True)
for name, value in results.items():
    print(f"{name}: {value:.4f}")

metadata = {
    "tensorflow": tf.__version__,
    "model": "EfficientNetB0",
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "threshold": 0.5,
    "classes": {"REAL": 0, "FAKE": 1},
    "test_metrics": {k: float(v) for k, v in results.items()},
}

with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model:", checkpoint_path)
print("Saved metadata:", MODEL_DIR / "model_metadata.json")
