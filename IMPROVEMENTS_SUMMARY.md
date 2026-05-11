# Code Improvements & Debugging Guide

## What Was Fixed

### 1. **Enhanced Preprocessing with Debug Output**
`preprocess_image()` now prints:
- Original image size and mode
- Shape after preprocessing
- Value ranges before/after normalization
- Final tensor properties

**Why:** Shows if preprocessing matches expectations

---

### 2. **Improved Model Prediction Debugging**
`get_prediction()` now outputs:
- Input tensor shape, dtype, and value range
- Raw model output values
- All class probabilities (not just argmax)
- Which class was predicted and why

**Why:** Reveals if model predictions make sense or are always the same

---

### 3. **Better Grad-CAM Computation**
`generate_gradcam()` now:
- Uses more robust layer detection (conv2d, conv, conv3d)
- Adds `tape.watch()` for gradient tracking
- Uses `training=False` parameter
- Shows all intermediate shapes and values
- Prints gradient ranges to detect if they're near-zero
- Detailed error reporting

**Why:** Identifies why Grad-CAM might be uniform or failing

---

### 4. **Debug Route for Single Image Testing**
New endpoint: `/debug/<filename>`

Usage:
```
1. Upload image: upload_image.jpg
2. Navigate to: http://localhost:5000/debug/upload_image.jpg
3. Check console for full diagnostic output
```

**Why:** Test without going through full prediction pipeline

---

## Critical Issues to Check

### ⚠️ Issue #1: Preprocessing Normalization

**Current code:**
```python
img_array = img_array / 255.0  # Normalizes to 0-1 range
```

**Might be wrong if your training used:**
- `(img / 127.5) - 1.0` → -1 to 1 range
- `from tensorflow.keras.applications.resnet50 import preprocess_input` → ImageNet normalization
- Custom mean/std normalization
- No normalization at all (0-255)

**Fix:** Find your training code and copy the exact preprocessing!

---

### ⚠️ Issue #2: Class Label Mapping

**Current code:**
```python
CLASS_NAMES = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Leaf Smut"
}
```

**Might be wrong!** Check your training data generator:

```python
# In your training script, find:
# Option A: directory-based naming
# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     'path/to/data',
#     labels='inferred'
# )
# Classes are sorted alphabetically: Brown Spot, Bacterial Blight, Leaf Smut
# So indices: 0=Brown, 1=Bacterial, 2=Leaf (NOT what's in CLASS_NAMES!)

# Option B: with explicit class names
# label_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# Option C: from ImageDataGenerator
# gen = ImageDataGenerator()
# data = gen.flow_from_directory(..., classes=['...', '...', '...'])
```

**Find the exact order used during training!**

---

### ⚠️ Issue #3: Identical Grad-CAM Heatmaps

**Likely causes:**
1. Model predicts same class for all images
   - Solution: Use `/debug` route on 3 different images, compare predictions
   
2. Preprocessing doesn't match training
   - Solution: Follow Issue #1 fix above

3. Gradients are near-zero
   - Look for: "Gradient value range: min=~0, max=~0"
   - Solution: Model may be poorly trained or needs fine-tuning

4. Model confidence is near 50%
   - Look for predictions like [0.33, 0.34, 0.33]
   - Solution: Model is uncertain; may need more training data

---

### ⚠️ Issue #4: Model Input Shape

**Your code assumes:**
```python
target_size=(224, 224)  # Width, Height
# With 3 color channels → (224, 224, 3)
```

**Verify this matches your model!**

```python
# In Flask environment:
from tensorflow import keras
model = keras.models.load_model('models/keras_cnn_model.h5')
print(model.input_shape)  # Should show (None, 224, 224, 3)

# If it shows something else:
# - (None, 256, 256, 3) → Change target_size=(256, 256)
# - (None, 299, 299, 3) → Change target_size=(299, 299)
```

---

## How to Diagnose

### Step 1: Run Debug Route
```bash
1. Start Flask: python app.py
2. Upload test image via web interface
3. Note the filename: e.g., "leaf_disease.jpg"
4. Navigate to: http://localhost:5000/debug/leaf_disease.jpg
5. Check the console output
```

### Step 2: Look for These Patterns

```
CHECK YOUR CONSOLE FOR PATTERNS:

Pattern 1: All images same class
  ✗ CNN prediction: Class 1, Confidence 92.00%
  ✗ ResNet prediction: Class 1, Confidence 91.00%
  → Preprocessing doesn't match, or class mapping is wrong

Pattern 2: Uniform heatmap (all same color)
  "Gradient value range: min=0.0000, max=0.0000"
  → Model predictions too similar, or model poorly trained

Pattern 3: Success
  ✓ CNN: Class 0, 78%, Class 1, 15%, Class 2, 7%
  ✓ ResNet: Class 1, 82%, Class 0, 10%, Class 2, 8%
  ✓ Gradient range: min=-0.05, max=0.08
  → Should work correctly
```

### Step 3: Cross-Check with Training

Find your training script and verify:
```python
# TRAINING SCRIPT:
train_images = tf.keras.preprocessing.image.load_img(
    path,
    target_size=(224, 224)  # ← Must match!
)
x = tf.keras.preprocessing.image.img_to_array(train_images)
x = x / 255.0  # ← Must match preprocessing!

# Or with ImageDataGenerator:
train_datagen = ImageDataGenerator(
    rescale=1./255  # ← This is the normalization
)

# Compare with app.py preprocessing_image()
```

---

## Most Likely Fixes (In Order)

1. **Fix preprocessing normalization**
   - Open your training script
   - Find the exact normalization line
   - Copy it to `preprocess_image()` in app.py

2. **Fix class label mapping**
   - Check order of classes in training data
   - Update `CLASS_NAMES` dict to match

3. **Verify model input shape**
   - Check if (224, 224) is correct
   - Adjust if needed

4. **If still wrong:**
   - Run `/debug` route
   - Paste the console output
   - Each log line tells you what's happening

---

## Files Modified

1. **app.py**
   - Enhanced `preprocess_image()` with debug output
   - Enhanced `get_prediction()` with comprehensive logging
   - Enhanced `generate_gradcam()` with better layer detection
   - Added `/debug/<filename>` route for testing
   - Added troubleshooting comments at top

2. **DIAGNOSIS_GUIDE.md** ← You are here
   - Complete diagnostic framework
   - Step-by-step testing procedures
   - Common solutions table

---

## Next Steps

1. **Test one image using `/debug` route**
2. **Compare console output with DIAGNOSIS_GUIDE.md**
3. **Identify which issue matches your symptoms**
4. **Apply the corresponding fix from Guide**
5. **Retest with `/debug` route to verify fix**

---

## Emergency Checklist

If predictions are completely wrong:

- [ ] Check console output from `/debug` route
- [ ] Verify preprocessing matches training code exactly
- [ ] Verify class mapping order matches training
- [ ] Confirm model files exist and load correctly
- [ ] Test with a clearly diseased leaf image
- [ ] Check that model has been trained (not just initialized)
- [ ] Verify image format is RGB (not BGR or grayscale)

Most issues (>80%) are fixed by correcting preprocessing or class mapping!
