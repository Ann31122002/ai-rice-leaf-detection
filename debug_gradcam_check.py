import os
from app import preprocess_image, generate_gradcam_simple, cnn_model

image_path = 'static/uploads/brown spot.jpg'
print('Testing:', image_path)
if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)

img_array = preprocess_image(image_path)
print('preprocess:', img_array is not None, 'shape', img_array.shape if img_array is not None else None)

heatmap = generate_gradcam_simple(cnn_model, img_array)
print('heatmap:', type(heatmap), heatmap.shape if heatmap is not None else heatmap)

if heatmap is not None:
    import cv2
    import numpy as np
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (224,224))
    hm = (heatmap*255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig,0.6,hm_color,0.4,0)
    out = 'static/gradcam/test_check.png'
    cv2.imwrite(out, overlay)
    print('saved', out, os.path.exists(out), os.path.getsize(out))
