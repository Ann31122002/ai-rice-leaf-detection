# Diagnosis Guide: Identical Grad-CAM & Incorrect Predictions

## Problems to Solve

1. **Grad-CAM heatmaps look identical for different images**
2. **Disease predictions are often incorrect**

---

## Root Cause Analysis Framework

### Issue 1: Identical Grad-CAM Heatmaps

This typically means:
- All images are producing the same prediction (same class)
- Or: Gradients are near-zero (uniform heatmap across all pixels)

**Possible causes:**
1. ❌ Model predicts same class regardless of input
2. ❌ Preprocessing doesn't match training (wrong normalization/format)
3. ❌ Model is not properly trained
4. ❌ Input shape doesn't match model expectations

---

### Issue 2: Incorrect Predictions

This occurs when:
- Disease classification is wrong even for obviously diseased leaves
- High confidence on incorrect classes

**Possible causes:**
1. ❌ Preprocessing doesn't match what model was trained on
2. ❌ Class mapping is wrong (class 0 ≠ what you think it is)
3. ❌ Wrong normalization method
4. ❌ Image orientation/format issues (RGB vs BGR)
5. ❌ Model weights are corrupted or for different task

---

## Step-by-Step Diagnosis

### Step 1: Test Preprocessing

Navigate to your Flask app's debug route with a test image:
```
http://localhost:5000/debug/test_image.jpg
```

**What to check in console output:**

```
=== PREPROCESSING DEBUG ===
Original image size: (640, 480)
Image mode: RGB
Resized to: (224, 224)
Array dtype before normalization: float32
Array shape: (224, 224, 3)
Array value range BEFORE norm: min=0.00, max=255.00
Array value range AFTER norm: min=0.0000, max=1.0000
Final shape with batch dimension: (1, 224, 224, 3)
Final dtype: float32
============================
```

✅ **Expected values:**
- dtype: float32
- Shape: (1, 224, 224, 3)
- After normalization: min ~0.0000, max ~1.0000

❌ **Bad signs:**
- Shape not (1, 224, 224, 3)  
- Values not in 0-1 range
- dtype not float32

---

### Step 2: Check Predictions

**Console output for predictions:**

```
=== CNN PREDICTION ===
Input shape: (1, 224, 224, 3)
Input dtype: float32
Input value range: min=0.0000, max=1.0000
Output type: <class 'numpy.ndarray'>
Output shape: (1, 3)
Output values (raw): [[0.15 0.65 0.20]]
Number of classes: 3
Class probabilities: [0.15 0.65 0.20]
Predicted class index: 1
Confidence: 65.00%
Class name: Brown Spot
========================
```

✅ **Expected:**
- Output shape (1, 3) for 3 classes
- Probabilities sum to ~1.0
- Each probability between 0-1
- Makes sense for the image

❌ **Red flags:**
- All images give same class
- Probabilities don't sum to 1
- Confidence always near 50%
- Class 0 predictions seem random

---

### Step 3: Verify Model Input Shape

Run this Python test in your Flask environment:

```python
from tensorflow import keras

cnn = keras.models.load_model('models/keras_cnn_model.h5')
print(f"CNN input shape: {cnn.input_shape}")
print(f"CNN output shape: {cnn.output_shape}")
print(f"Number of parameters: {cnn.count_params()}")

resnet = keras.models.load_model('models/keras_resnet_finetuned.h5')
print(f"ResNet input shape: {resnet.input_shape}")
print(f"ResNet output shape: {resnet.output_shape}")
```

**Expected output:**
```
CNN input shape: (None, 224, 224, 3)
CNN output shape: (None, 3)
ResNet input shape: (None, 224, 224, 3)
ResNet output shape: (None, 3)
```

If input shape is NOT (None, 224, 224, 3), you need to adjust preprocessing!

---

### Step 4: Verify Preprocessing Matches Training

**This is critical!** Check your training code:

```python
# Need to find your training script and check these lines:
# Usually in: data_augmentation, preprocessing, model input

# Example A: Simple normalization
x_train = x_train / 255.0  # YOUR CODE TOO?

# Example B: ImageNet preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
x_train = preprocess_input(x_train)  # Different! Uses mean-std

# Example C: Custom range  
x_train = (x_train / 127.5) - 1.0  # -1 to 1 range

# Example D: Per-channel normalization
mean = [0.5, 0.5, 0.5]
std = [0.2, 0.2, 0.2]
x_train = (x_train - mean) / std
```

**Copy preprocessing from your training code!** Mismatch here is the #1 cause of incorrect predictions.

---

### Step 5: Check Class Mapping

Look at your training code to find what each class index means:

```python
# From training code - find class definitions:
CLASS_NAMES = {
    0: "Class A",
    1: "Class B",  
    2: "Class C"
}

# These are in order of your training dataset
# If training used alphabetical order: Brown Spot, Leaf Smut, Bacterial Blight
# Then indices might be: 0=Brown, 1=Leaf, 2=Bacterial
# Not: 0=Bacterial, 1=Brown, 2=Leaf
```

Verify the order matches your training!

---

### Step 6: Check Grad-CAM Layer

**Console output:**

```
=== GRADCAM GENERATION ===
Input shape: (1, 224, 224, 3)
✓ Using layer for GradCAM: conv2d_last_or_whatever
  Layer output shape: (7, 7, 512)
Predictions: [[0.15 0.65 0.20]]
Conv outputs shape: (1, 7, 7, 512)
Predicted class index: 1
Predicted value: 0.65
✓ Gradients computed successfully
  Gradient shape: (1, 7, 7, 512)
  Gradient value range: min=-0.0324, max=0.0127
Pooled gradients shape: (512,)
Heatmap shape: (7, 7)
Heatmap value range (after norm): min=0.0000, max=1.0000
✓ Final heatmap shape: (224, 224)
  Final heatmap value range: min=0.0000, max=1.0000
=========================
```

✅ **Good signs:**
- Layer found successfully
- Gradients computed (not None)
- Heatmap has variation (min=0, max=1)
- Final shape is (224, 224)

❌ **Problems:**
- No convolution layer found → check layer names
- "Gradients are None" → model not trainable
- Gradient range is tiny (min/max ~0) → uniform heatmap
- Heatmap value range all 0s → no activation differences

---

## Quick Fix Checklist

### If Predictions Are Wrong:

1. **Check normalization:**
   ```python
   # In your training code, find this line:
   img_normalized = img / 255.0  # or (img / 127.5) - 1 or preprocess_input()
   # Use SAME in preprocess_image()
   ```

2. **Verify class mapping:**
   ```python
   # Print training class order:
   print(train_dataset.class_names)  # From training
   # Update CLASS_NAMES in app.py to match exactly
   ```

3. **Check image shape:**
   ```python
   # Verify (224, 224) is correct for your model
   # Some ResNets use (256, 256) or (299, 299)
   ```

### If Grad-CAM Is Uniform:

1. **Check if model is making different predictions:**
   - Use `/debug/` route with 3-4 different images
   - Do predictions change? If not → model issue

2. **Check gradient computation:**
   - Look for "Gradient range: min=X, max=X" in console
   - If values are tiny or all zeros → model may be poorly trained

3. **Try image with obvious features:**
   - Use image with clear disease symptoms
   - Check if predictions change dramatically

---

## Files to Check

1. **Your training code** (find it!)
   - How was preprocessing done?
   - What's class label order?
   - What input size?

2. **Model files**
   - `models/keras_cnn_model.h5` - exists?
   - `models/keras_resnet_finetuned.h5` - exists?

3. **App code**
   - `preprocess_image()` - matches training?
   - `CLASS_NAMES` - correct order?
   - Layer detection working?

---

## Testing Commands

Run these in Flask shell (`flask shell`):

```python
# Test 1: Model loading
from app import cnn_model, resnet_model
print(f"CNN loaded: {cnn_model is not None}")
print(f"ResNet loaded: {resnet_model is not None}")

# Test 2: Single image prediction
from app import preprocess_image, get_prediction
img = preprocess_image('static/uploads/test.jpg')
class_idx, conf = get_prediction(img, 'cnn')
print(f"Prediction: Class {class_idx}, Confidence {conf}%")

# Test 3: Check model structure
print(cnn_model.summary())
```

---

## Common Solutions

| Problem | Solution |
|---------|----------|
| All images → same class | Check preprocessing mismatches; test with `/debug` route |
| Always class 0 | Wrong class mapping order; check training label order |
| Low confidence (<55%) | Model might be poorly trained or needs fine-tuning |
| Uniform heatmap | Gradients are zero; model may not be differentiable or predictions are too certain |
| Wrong shape error | Verify model expects (224, 224), not (256, 256) or other |
| "No conv layer found" | Check model architecture; try different layer names |

---

## Still Stuck?

Verify this specific workflow works:

```bash
1. Start Flask app
2. Upload test image
3. Check console for preprocessing output
4. Navigate to /debug/test_image.jpg  
5. Compare console output with expectations above
6. Paste console output here to diagnose
```

The debug output usually reveals the exact issue!
