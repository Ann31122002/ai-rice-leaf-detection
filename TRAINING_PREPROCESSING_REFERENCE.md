# Training Code Preprocessing - What to Look For

## Goal
Find your training code and match the preprocessing **exactly** in Flask app.

---

## Common Training Preprocessing Patterns

### Pattern A: Simple 0-255 to 0-1 Normalization (Most Common)

**Training code:**
```python
# Method 1: In ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,      # ← THIS! Means divide by 255
    rotation_range=20,
    # ... other augmentations
)
```

**Or Method 2: Direct preprocessing**
```python
# Directly in data loading
X_train = np.array(images) / 255.0  # ← THIS!
```

**Or Method 3: TensorFlow datasets**
```python
def preprocess(image, label):
    image = image / 255.0  # ← THIS!
    return image, label

train_dataset = train_dataset.map(preprocess)
```

**Corresponding Flask code (ALREADY CORRECT):**
```python
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # ✓ Matches!
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

✅ **Status:** CORRECT - Your Flask code matches this!

---

### Pattern B: ImageNet Preprocessing (ResNet, VGG, etc.)

**Training code:**
```python
# Method 1: Using Keras preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input

# In data loading:
x_train = preprocess_input(x_train)
```

**Or Method 2: Manual ImageNet normalization**
```python
# ImageNet mean and std
mean = np.array([103.939, 116.779, 123.68])  # BGR order!
x_train = x_train - mean
# Or with RGB:
mean_rgb = np.array([123.68, 116.779, 103.939])
x_train = x_train - mean_rgb
```

**Or Method 3: In data pipeline**
```python
def imagenet_preprocess(image, label):
    image = image - [123.68, 116.779, 103.939]
    return image, label
```

**Corresponding Flask fix needed:**
```python
# CHANGE preprocess_image() to:
def preprocess_image(image_path, target_size=(224, 224)):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    # img_array = img_array / 255.0  # ✗ REMOVE THIS
    img_array = preprocess_input(img_array)  # ✓ ADD THIS
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

⚠️ **Status:** If your code uses ResNet, this might be the issue!

---

### Pattern C: -1 to 1 Range Normalization

**Training code:**
```python
# Method 1: Direct scaling
X_train = (X_train / 127.5) - 1.0  # Range -1 to 1

# Method 2: In preprocessing function
def preprocess(img):
    return (img / 127.5) - 1.0

# Method 3: In ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./127.5,  # This means divide by 127.5
    # ... plus you need to handle the -1 part
)
```

**Corresponding Flask fix:**
```python
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0  # ✓ This instead of / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

⚠️ **Status:** Check if your training uses this!

---

### Pattern D: Per-Channel Mean/Std Subtraction

**Training code:**
```python
# Method 1: Using standard ImageNet stats
mean = [0.485, 0.456, 0.406]  # For 0-1 normalized images
std = [0.229, 0.224, 0.225]

# Apply normalization
for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]

# Method 2: Using TorchVision style
from torchvision.transforms import Normalize
normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Method 3: Custom per-dataset stats
# These should be computed from YOUR training set!
mean = [0.5, 0.4, 0.3]  # Your custom values
std = [0.2, 0.2, 0.2]   # Your custom values
```

**Corresponding Flask fix:**
```python
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    
    # First normalize to 0-1
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Then apply mean/std (copy from your training!)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

⚠️ **Status:** Check if your training uses this!

---

### Pattern E: No Normalization (Raw 0-255)

**Training code:**
```python
# Rare but possible
X_train = np.array(images)  # No division, stays 0-255

# Or explicitly
X_train = X_train / 1.0  # No-op, stays 0-255
```

**Corresponding Flask fix:**
```python
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    
    img_array = np.array(img, dtype=np.float32)
    # NO normalization - use as-is with 0-255 range
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

⚠️ **Status:** Unlikely but check if mentioned in training code!

---

## How to Find Your Preprocessing

### Step 1: Locate Training Script
Look for files named:
- `train.py`
- `train_model.py`
- `training.py`
- `train_*.ipynb` (Jupyter notebooks)
- `fit_model.py`
- Inside `models/` or `notebooks/` directory

### Step 2: Search for Keywords
In your training script, look for these lines:

```python
# Search for patterns like:
rescale=1./255        # ImageDataGenerator
rescale=1./127.5      # Range -1 to 1
preprocess_input      # ImageNet preprocessing
/ 255.0               # Manual 0-1 norm
/ 127.5 - 1           # Manual -1 to 1 norm
- mean / std          # Per-channel normalization
Normalize(            # Torchvision normalize
ImageDataGenerator    # Check rescale parameter
flow_from_directory   # Check preprocessing_function
```

### Step 3: Copy Exact Preprocessing

When you find the preprocessing, copy the **exact** code to Flask.

**Example:**
```python
# FROM TRAINING CODE:
train_datagen = ImageDataGenerator(
    rescale=1./255,  # ← Copy this
    rotation_range=20,
    horizontal_flip=True,
)

# TO FLASK CODE (app.py):
def preprocess_image(...):
    img_array = np.array(img) / 255.0  # ✓ Copied from rescale=1./255
```

---

## Size/Shape Verification

### Step 1: Check Training Image Size
```python
# In your training code, find:
img = tf.image.resize(img, [224, 224])  # Check the size!
# or
img = cv2.resize(img, (224, 224))
# or  
img = img.resize((224, 224))
```

**Common sizes:**
- ResNet, VGG, Inception, MobileNet: **224×224**
- InceptionV3, InceptionResNetV2: **299×299**
- Some custom models: **256×256** or **512×512**

### Step 2: Update Flask if Needed
```python
# In app.py, if training used different size:
def preprocess_image(image_path, target_size=(256, 256)):  # ← Change this!
    # ... rest of code
```

### Step 3: Verify All Models Use Same Size
```python
# Check both models in training:
cnn_model.input_shape      # Should show same size for both
resnet_model.input_shape   # Should match!
```

---

## Class Label Order Verification

### Method 1: From ImageDataGenerator
```python
# TRAINING CODE:
train_generator = datagen.flow_from_directory(
    'path/to/train',
    target_size=(224, 224),
    classes=['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut'],
    class_mode='categorical'
)

# CLASS ORDER:
# Index 0 = Bacterial Leaf Blight
# Index 1 = Brown Spot
# Index 2 = Leaf Smut

# Copy to app.py:
CLASS_NAMES = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Leaf Smut"
}
```

### Method 2: Alphabetical (Default)
```python
# If no explicit class list provided:
train_generator = datagen.flow_from_directory(
    'path/to/train'
    # Classes auto-detected from folder names
)

# Classes are sorted ALPHABETICALLY!
# Folders: bacterial_blight/, brown_spot/, leaf_smut/
# Order: bacterial_blight < brown_spot < leaf_smut
# So indices: 0=bacterial, 1=brown, 2=leaf

# Copy to app.py:
CLASS_NAMES = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Leaf Smut"
}
```

### Method 3: From Numpy/TensorFlow
```python
# TRAINING CODE:
class_names = ['Class A', 'Class B', 'Class C']
# or
class_names = np.array(['...', '...', '...'])

# Copy the order to app.py CLASS_NAMES!
```

### Method 4: Print During Training
```python
# Add this to training code to print mapping:
for batch, labels in train_generator:
    print(train_generator.class_indices)
    break
# Output: {'class_name1': 0, 'class_name2': 1, ...}

# Use this exact mapping in app.py
```

---

## Quick Reference Table

| Pattern | Key Line | Flask Fix |
|---------|----------|-----------|
| **A** (0-1 norm) | `rescale=1./255` | `img_array / 255.0` ✓ Already correct |
| **B** (ImageNet) | `preprocess_input()` | `preprocess_input(img_array)` ✗ Need to add |
| **C** (-1 to 1) | `(img / 127.5) - 1.0` | `(img_array / 127.5) - 1.0` ✗ Need to change |
| **D** (Mean/Std) | `(x - mean) / std` | Add mean/std loop ✗ Need to add |
| **E** (No norm) | `# No rescale` | Remove division ✗ Need to change |

---

## Testing Your Fix

After updating preprocessing in Flask:

1. **Start Flask app**
2. **Upload test image**
3. **Check console output for line:**
   ```
   Array value range AFTER norm: min=X.XXX, max=Y.YYY
   ```

4. **Compare with training:**
   - Pattern A: min≈0, max≈1 ✓
   - Pattern B: min≈-2, max≈2 or min≈-1.5, max≈2 ✓
   - Pattern C: min≈-1, max≈1 ✓
   - Pattern D: min≈-3, max≈3 (varies) ✓
   - Pattern E: min≈0, max≈255 ✓

5. **If values don't match training:**
   - You found the problem!
   - Fix and retest

---

## Still Can't Find Training Code?

If you lost the training script:

1. **Check model metadata:**
   ```python
   model = keras.models.load_model('path/to/model.h5')
   print(model.input_shape)    # Might show expected input
   print(model.get_config())   # Full config
   ```

2. **Try Pattern A first** (most common)
   - If that doesn't work, try B, then C, etc.

3. **Use `/debug` route** to test each pattern:
   - Change preprocessing in app.py
   - Use `/debug/<image>` to see results
   - Check console for accuracy

4. **Test predictions manually:**
   ```bash
   # Flask shell
   from app import preprocess_image, get_prediction, cnn_model
   
   img = preprocess_image('test.jpg')
   class_idx, conf = get_prediction(img, 'cnn')
   print(f"Class: {class_idx}, Confidence: {conf}%")
   
   # Try on obviously diseased leaf
   # If it predicts wrong disease or confidence <60%, wrong preprocessing!
   ```

---

## Summary

**Most important:** Find training preprocessing and copy **exactly**.

Run `/debug/<image>` after changes to verify predictions improve!
