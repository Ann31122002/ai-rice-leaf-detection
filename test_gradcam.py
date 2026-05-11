#!/usr/bin/env python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

print("="*60)
print("TESTING GRADCAM GENERATION")
print("="*60)

# Load model
print("\n1. Loading model...")
model = keras.models.load_model('models/keras_cnn_model.h5')
print("✓ Model loaded")

# Load test image
print("\n2. Loading test image...")
img_path = 'static/uploads/DSC_0113.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"✗ Failed to load image from {img_path}")
    exit(1)
print(f"✓ Image loaded. Original shape: {img.shape}")

# Preprocess
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img_orig = img.copy()
img = img.astype(np.float32) / 255.0
img_array = np.expand_dims(img, axis=0)
print(f"✓ Image preprocessed. Final shape: {img_array.shape}")

# Get prediction
print("\n3. Getting prediction...")
pred = model.predict(img_array, verbose=0)
print(f"✓ Prediction: {pred}")
pred_class = np.argmax(pred[0])
pred_conf = float(np.max(pred[0])) * 100
print(f"  Predicted class: {pred_class}, Confidence: {pred_conf:.2f}%")

# Find conv layer
print("\n4. Finding convolutional layer...")
conv_name = None
for layer in reversed(model.layers):
    print(f"  - Checking layer: {layer.name} (type: {type(layer).__name__})")
    if 'conv' in layer.name.lower():
        conv_name = layer.name
        print(f"    ✓ Found Conv2D layer!")
        break

if not conv_name:
    print("✗ No Conv2D layer found!")
    exit(1)

# Try GradCAM
print(f"\n5. Generating GradCAM using {conv_name}...")
try:
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Get conv layer
    conv_layer = model.get_layer(conv_name)
    print(f"✓ Conv layer shape: {conv_layer.output.shape}")
    
    # Create grad model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.output, conv_layer.output]
    )
    print("✓ Grad model created")
    
    # Forward pass with gradients
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img_tensor, training=False)
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[0, class_idx]
        print(f"✓ Forward pass done. Target class: {class_idx.numpy()}, score: {class_channel.numpy():.4f}")
    
    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        print("✗ Gradients are None!")
        exit(1)
    
    print(f"✓ Gradients obtained! Shape: {grads.shape}")
    
    # Reduce mean
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
    print(f"✓ Pooled gradients shape: {pooled_grads.shape}")
    
    # Generate heatmap
    conv_np = conv_outputs[0].numpy()
    grads_np = pooled_grads.numpy()
    
    heatmap = np.zeros((conv_np.shape[0], conv_np.shape[1]), dtype=np.float32)
    for i in range(conv_np.shape[2]):
        heatmap += grads_np[i] * conv_np[:, :, i]
    
    print(f"✓ Heatmap computed. Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    # Apply ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    # Resize
    heatmap = cv2.resize(heatmap, (224, 224))
    print(f"✓ Heatmap resized and normalized. Final range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    # Create overlay
    print("\n6. Creating overlay...")
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    img_orig_uint8 = (img_orig).astype(np.uint8)
    alpha = 0.4
    overlay = cv2.addWeighted(img_orig_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    print(f"✓ Overlay created. Shape: {overlay.shape}")
    
    # Save
    os.makedirs('static/gradcam', exist_ok=True)
    output_path = 'static/gradcam/test_gradcam.png'
    success = cv2.imwrite(output_path, overlay)
    
    if success and os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"✓ Saved to {output_path} ({size} bytes)")
    else:
        print(f"✗ Failed to save")
        exit(1)
    
    print("\n" + "="*60)
    print("✓ GRADCAM TEST SUCCESSFUL!")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
