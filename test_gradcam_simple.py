#!/usr/bin/env python
import sys
import os
sys.path.append('.')

from app import generate_gradcam_simple, preprocess_image, cnn_model
import numpy as np

print("Testing GradCAM generation...")

# Test with an existing image
filepath = 'static/uploads/DSC_0113.jpg'
print(f"Testing with image: {filepath}")

if not os.path.exists(filepath):
    print(f"ERROR: Image file does not exist: {filepath}")
    sys.exit(1)

img_array = preprocess_image(filepath)
if img_array is not None:
    print(f'✓ Image preprocessed successfully, shape: {img_array.shape}')

    # Test GradCAM generation
    heatmap = generate_gradcam_simple(cnn_model, img_array)
    if heatmap is not None:
        print(f'✓ GradCAM generated successfully, shape: {heatmap.shape}')
        print(f'✓ Heatmap range: {heatmap.min():.3f} to {heatmap.max():.3f}')
        print("SUCCESS: GradCAM is working!")
    else:
        print('✗ GradCAM generation failed')
        sys.exit(1)
else:
    print('✗ Image preprocessing failed')
    sys.exit(1)