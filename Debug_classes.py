"""
Run this ONCE with:  python debug_classes.py
It prints the exact class index order your retrained model expects,
so you can correct CNN_CLASS_NAMES in app.py.
"""
import os, sys

# ── 1. Show what Keras would assign alphabetically ──────────────────────────
DATASET_DIR = 'dataset/train'   # ← change to YOUR training folder path

if os.path.isdir(DATASET_DIR):
    folders = sorted(os.listdir(DATASET_DIR))   # Keras sorts alphabetically
    print("\n=== Keras alphabetical class index order ===")
    for idx, name in enumerate(folders):
        print(f"  {idx} → {name}")
else:
    print(f"WARNING: '{DATASET_DIR}' not found. Set DATASET_DIR to your training folder.")
    print("Manually list your training subfolders sorted A→Z to determine the order.")

# ── 2. Load model and probe output shape ────────────────────────────────────
print("\n=== Loading model ===")
try:
    import tensorflow as tf
    try:
        import keras
    except ImportError:
        from tensorflow import keras

    model = keras.models.load_model('models/keras_cnn_model.h5', compile=False)
    print(f"Input  shape : {model.input_shape}")
    print(f"Output shape : {model.output_shape}  ← number of classes = {model.output_shape[-1]}")

    import numpy as np
    dummy = np.zeros((1, 128, 128, 3), dtype='float32')
    out = model(dummy, training=False).numpy()
    print(f"Dummy output : {out[0]}  (should sum ≈ 1.0 for softmax)")

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback; traceback.print_exc()

print("""
─────────────────────────────────────────────────────────────────
ACTION: Copy the alphabetical folder order printed above and make
sure CNN_CLASS_NAMES in app.py matches it EXACTLY, e.g.:

CNN_CLASS_NAMES = {
    0: "Bacterial Leaf Blight",   # folder: bacterial_leaf_blight
    1: "Brown Spot",              # folder: brown_spot
    2: "Healthy Rice Leaf",       # folder: healthy_rice_leaf
    3: "Leaf Blast",              # folder: leaf_blast
}

If the folder names sort differently on your machine, update accordingly.
─────────────────────────────────────────────────────────────────
""")