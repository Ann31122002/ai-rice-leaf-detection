# Quick Start: Debugging Grad-CAM & Predictions - IMMEDIATE STEPS

## You Have 3 Minutes? Do This:

### Step 1: Start Flask (30 seconds)
```bash
# Terminal
cd c:\Users\User\Downloads\Nexa
python app.py
```

Wait for: `Running on http://127.0.0.1:5000`

### Step 2: Upload a Test Image (30 seconds)
1. Open browser: http://localhost:5000
2. Click "Upload"
3. Select a clearly diseased leaf
4. Click "Preview"

### Step 3: Use Debug Route (2 minutes)
1. After upload, note the filename shown, e.g., `test_image.jpg`
2. Open new browser tab: `http://localhost:5000/debug/test_image.jpg`
3. **Check the console** (where you ran `python app.py`)

### Step 4: Read Console Output (1 minute)

**Look for these sections:**

```
=== PREPROCESSING DEBUG ===
Array value range AFTER norm: min=X.XXX, max=Y.YYY
```

**CRITICAL:** Is this the right range?
- Should be: `min=0.0000, max=1.0000` (if using 0-1 normalization)
- Or: `min=-1.0000, max=1.0000` (if using -1 to 1)
- Or: `min=0, max=255` (if no normalization)

If values are weird → you found problem #1!

```
=== CNN PREDICTION ===
Class probabilities: [X, Y, Z]
Predicted class index: N
```

**CRITICAL:** Do different images give different predictions?
- Same class for all images? → Problem #2!
- Different classes? → Good, continue testing

```
=== GRADCAM GENERATION ===
Gradient value range: min=X, max=X
```

**CRITICAL:** Is this range near zero?
- `min=0.0000, max=0.0000` or very tiny? → Uniform heatmap (Problem #3!)
- `min=-0.05, max=0.08` or similar? → Looks good

---

## What Each Problem Means

### Problem #1: Wrong Preprocessing Range
**Symptom:** Preprocessing values don't match what training used
```
Expected: min≈0, max≈1
Got: min≈0, max≈255
```

**Fix:** Edit `preprocess_image()` in app.py
- Find your training code
- Copy preprocessing from there
- Paste into Flask app

**Docs:** See `TRAINING_PREPROCESSING_REFERENCE.md`

---

### Problem #2: All Images Same Class
**Symptom:** Every image predicts same disease
```
Image 1: Class 1 (Brown Spot)
Image 2: Class 1 (Brown Spot)  
Image 3: Class 1 (Brown Spot)
```

**Likely causes:**
1. Class mapping is wrong (indices reversed)
2. Preprocessing doesn't match training
3. Model weights are corrupted

**Quick fix:**
1. Try different test images
2. If still same class → check preprocessing (Problem #1)
3. If different classes sometimes → fine, but may need class mapping fix

**Docs:** See `DIAGNOSIS_GUIDE.md` section "Check Class Mapping"

---

### Problem #3: Uniform Grad-CAM Heatmap
**Symptom:** Grad-CAM looks same for all images (all one color)
```
Gradient value range: min=0.0000, max=0.0000
```

**Causes:**
1. Model predictions identical (Problem #2)
2. Model not well-trained
3. Gradients actually zero (rare)

**Fix:** First solve Problem #2, then retest

---

## After Finding Problem

### 1. Check DIAGNOSIS_GUIDE.md
Links to detailed explanation with examples

### 2. Check TRAINING_PREPROCESSING_REFERENCE.md
Shows exactly how to copy preprocessing from training code

### 3. Make One Change
Don't change multiple things at once!
- Change only preprocessing, OR
- Change only class mapping, OR
- Change only image size

### 4. Retest with /debug Route
```
http://localhost:5000/debug/test_image.jpg
```

### 5. Check Console Output Again
Did the values improve?

### 6. Repeat Until Fixed

---

## Most Common Fix (90% of Cases)

**Copy preprocessing from training code:**

1. Open your training script
2. Find line with: `rescale`, `preprocess_input`, `/255`, `/127.5`, or `mean`/`std`
3. Find the same pattern in `preprocess_image()` in app.py
4. Update to match exactly
5. Test with `/debug` route

**That's it!** Most prediction problems are preprocessing mismatches.

---

## If You Get Stuck

### Immediate Help
1. Check console output from `/debug` route
2. Find which section doesn't look right
3. Go to DIAGNOSIS_GUIDE.md
4. Find matching symptom
5. Follow the fix

### Still Stuck?
1. Run `/debug/<image>` with 3 different images
2. Copy the **full console output**
3. Check against DIAGNOSIS_GUIDE.md sections:
   - "Input value range" should match training
   - "Class probabilities" should be different for different images
   - "Gradient range" should not be all zeros

### Last Resort
Test preprocessing manually:
```python
# Flask shell
from app import preprocess_image
import numpy as np

img = preprocess_image('static/uploads/test.jpg')
print(f"Shape: {img.shape}")
print(f"Min: {img.min()}, Max: {img.max()}")
print(f"Mean: {img.mean()}, Std: {img.std()}")

# Compare with what training used
```

---

## File Reference

| File | Purpose | When to Read |
|------|---------|-------------|
| **DIAGNOSIS_GUIDE.md** | Detailed troubleshooting with examples | When /debug shows problem |
| **IMPROVEMENTS_SUMMARY.md** | What was changed in code | To understand new features |
| **TRAINING_PREPROCESSING_REFERENCE.md** | How to find & copy training preprocessing | When fixing Problem #1 |
| **app.py** | The Flask application with debugging | To make fixes |

---

## One More Time: The Plan

1. **Run Flask** → `python app.py`
2. **Upload image** → select `.jpg` or `.png`
3. **Debug image** → navigate to `/debug/filename.jpg`
4. **Check console** → find the 3 "===" sections
5. **Compare values** → do they match training?
6. **Fix mismatches** → update app.py
7. **Retest** → check console again
8. **Repeat** until predictions look correct

---

## Success Criteria

When fixed, you should see:

1. **Different predictions for different images**
   ```
   Healthy leaf: Class 0, Confidence 85%
   Diseased leaf: Class 1, Confidence 92%
   ```

2. **Varied Grad-CAM heatmaps**
   - Different colors/intensity for different images
   - Red areas match actual disease symptoms

3. **Correct disease identification**
   - Clearly diseased leaves → detected as diseased
   - Healthy leaves → detected as healthy
   - Confidence typically 75%+ for good images

---

## Still Loading? Give It Time

Grad-CAM computation takes several seconds (5-15 sec normal).
Check for `✓ GradCAM overlay saved:` in console.

If stuck for >30 seconds:
1. Check if Flask is still running
2. Check if browser shows any errors
3. Try a smaller image
4. Check console for error messages

---

**You got this!** Most issues are just preprocessing mismatches. The `/debug` route will show you exactly what's wrong.
