# # from flask import Flask, render_template, request, redirect, url_for, session
# # import os
# # import numpy as np
# # from PIL import Image
# # import cv2

# # app = Flask(__name__)
# # app.secret_key = 'your_secret_key_here'

# # # ---------------------------------------------------------------------------
# # # Folder configuration
# # # ---------------------------------------------------------------------------
# # UPLOAD_FOLDER = 'static/uploads'
# # GRADCAM_FOLDER = 'static/gradcam'
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# # GRADCAM_OUTPUT_SIZE = (512, 512)

# # # ---------------------------------------------------------------------------
# # # Class label mappings
# # #
# # # Keras ImageDataGenerator.flow_from_directory() assigns indices by sorting
# # # training folder names ALPHABETICALLY (case-insensitive on most OS).
# # #
# # # Run `python debug_classes.py` to confirm your exact folder sort order.
# # #
# # # Common folder names → alphabetical sort → index:
# # #   bacterial_leaf_blight → 0
# # #   brown_spot            → 1
# # #   healthy_rice_leaf     → 2
# # #   leaf_blast            → 3
# # #
# # # ⚠️  IF Leaf Blast predictions are WRONG, the old model had index 2 = leaf_smut.
# # #     After retraining with 4 classes the new index 2 = healthy_rice_leaf and
# # #     index 3 = leaf_blast.  Make sure your NEW model file is the one being
# # #     loaded (check models/keras_cnn_model.h5 file date).
# # # ---------------------------------------------------------------------------

# # CNN_CLASS_NAMES = {
# #     0: "Bacterial Leaf Blight",
# #     1: "Brown Spot",
# #     2: "Healthy Rice Leaf",
# #     3: "Leaf Blast",
# # }

# # STATUS_MESSAGES = {
# #     0: (
# #         "Bacterial Leaf Blight detected. This is a serious bacterial disease. "
# #         "Recommended actions: Remove and destroy affected leaves immediately, "
# #         "apply copper-based bactericides (e.g. copper oxychloride), improve field "
# #         "drainage, avoid overhead irrigation, and use resistant varieties."
# #     ),
# #     1: (
# #         "Brown Spot detected. This fungal disease is often linked to nutrient "
# #         "deficiency. Recommended actions: Remove infected leaves, apply fungicide "
# #         "(e.g. mancozeb or propiconazole), improve soil nutrition especially "
# #         "potassium and silicon, reduce plant density for better air circulation."
# #     ),
# #     2: (
# #         "Healthy Rice Leaf detected. The plant appears healthy with no visible "
# #         "signs of disease. Continue good agricultural practices: proper irrigation, "
# #         "balanced NPK fertilisation, and regular field monitoring to catch any "
# #         "early disease signs."
# #     ),
# #     3: (
# #         "Leaf Blast detected. This is one of the most destructive rice diseases. "
# #         "Recommended actions: Remove and destroy infected plant parts, apply "
# #         "systemic fungicide (e.g. tricyclazole or isoprothiolane), avoid excessive "
# #         "nitrogen fertilisation, improve air circulation, and consider resistant "
# #         "varieties for future planting."
# #     ),
# # }

# # # ---------------------------------------------------------------------------
# # # Lazy-loaded TensorFlow globals
# # # ---------------------------------------------------------------------------
# # tf     = None
# # keras  = None
# # cnn_model = None


# # # ---------------------------------------------------------------------------
# # # Preprocessing  (must match training exactly)
# # # ---------------------------------------------------------------------------
# # def preprocess_for_cnn(image_path, target_size=(128, 128)):
# #     """
# #     Replicates Keras ImageDataGenerator(rescale=1/255) + flow_from_directory:
# #       - PIL RGB open  (no BGR swap)
# #       - LANCZOS resize (matches Keras/PIL default)
# #       - divide by 255
# #     """
# #     img = Image.open(image_path).convert('RGB')
# #     img = img.resize(target_size, Image.LANCZOS)
# #     arr = np.array(img, dtype=np.float32) / 255.0
# #     return np.expand_dims(arr, axis=0)   # (1, H, W, 3)


# # # ---------------------------------------------------------------------------
# # # CNN prediction
# # # ---------------------------------------------------------------------------
# # def get_prediction(image_array):
# #     """Return (class_idx, confidence_percent) or (None, None) on error."""
# #     try:
# #         if cnn_model is None:
# #             print("ERROR: CNN model not loaded")
# #             return None, None

# #         img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# #         output = cnn_model(img_tensor, training=False).numpy()
# #         print(f"[cnn] raw output: {output[0]}")

# #         if output.shape[1] > 1:                        # softmax (4-class)
# #             class_idx  = int(np.argmax(output[0]))
# #             confidence = float(np.max(output[0])) * 100
# #         else:                                           # sigmoid (binary)
# #             prob       = float(output[0][0])
# #             class_idx  = 1 if prob >= 0.5 else 0
# #             confidence = max(prob, 1.0 - prob) * 100

# #         label = CNN_CLASS_NAMES.get(class_idx, "Unknown")
# #         print(f"[cnn] class_idx={class_idx}  label='{label}'  confidence={confidence:.2f}%")
# #         return class_idx, confidence

# #     except Exception as e:
# #         print(f"Error in get_prediction: {e}")
# #         import traceback; traceback.print_exc()
# #         return None, None


# # # ---------------------------------------------------------------------------
# # # GradCAM helpers
# # # ---------------------------------------------------------------------------
# # def _collect_conv_layers(model, _seen=None):
# #     if _seen is None:
# #         _seen = set()
# #     results = []
# #     for layer in model.layers:
# #         if id(layer) in _seen:
# #             continue
# #         _seen.add(id(layer))
# #         if hasattr(layer, 'layers'):
# #             results.extend(_collect_conv_layers(layer, _seen))
# #         elif type(layer).__name__ == 'Conv2D':
# #             results.append((layer, model))
# #     return results


# # def _find_gradcam_conv_layer(model):
# #     COLLAPSES = {'GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'Flatten'}

# #     all_conv = _collect_conv_layers(model)
# #     if not all_conv:
# #         return None

# #     def _flat(m, seen=None):
# #         if seen is None:
# #             seen = set()
# #         out = []
# #         for l in m.layers:
# #             if id(l) in seen:
# #                 continue
# #             seen.add(id(l))
# #             if hasattr(l, 'layers'):
# #                 out.extend(_flat(l, seen))
# #             else:
# #                 out.append(l)
# #         return out

# #     flat = _flat(model)
# #     collapse_idx = next(
# #         (i for i, l in enumerate(flat) if type(l).__name__ in COLLAPSES), None
# #     )

# #     if collapse_idx is None:
# #         layer, owner = all_conv[-1][0], all_conv[-1][1]
# #         print(f"GradCAM: no collapse layer, using last conv '{layer.name}'")
# #         return owner, layer.name

# #     best = None
# #     for layer, owner in all_conv:
# #         try:
# #             idx = flat.index(layer)
# #         except ValueError:
# #             continue
# #         if idx < collapse_idx:
# #             best = (owner, layer.name)
# #         else:
# #             break

# #     if best is None:
# #         best = (all_conv[-1][1], all_conv[-1][0].name)

# #     print(f"GradCAM: target conv='{best[1]}' in owner='{best[0].name}'")
# #     return best


# # def generate_gradcam_simple(model, image_array, target_size=GRADCAM_OUTPUT_SIZE):
# #     try:
# #         print("\n=== GRADCAM ===")
# #         if image_array is None:
# #             return None

# #         img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# #         if len(img_tensor.shape) == 3:
# #             img_tensor = tf.expand_dims(img_tensor, axis=0)

# #         _ = model(img_tensor, training=False)   # warm-up

# #         result = _find_gradcam_conv_layer(model)
# #         if result is None:
# #             print("ERROR: no Conv2D layer found")
# #             return None
# #         owner_model, conv_name = result

# #         # Build feature extractor
# #         try:
# #             conv_layer = owner_model.get_layer(conv_name)
# #             extractor  = tf.keras.models.Model(
# #                 inputs=owner_model.inputs,
# #                 outputs=[conv_layer.output, owner_model.output],
# #                 name="gradcam_extractor"
# #             )
# #         except Exception as e:
# #             print(f"Functional extractor failed ({e}), using Sequential chaining...")
# #             inp = tf.keras.Input(shape=owner_model.input_shape[1:])
# #             x, conv_out_tensor = inp, None
# #             start = 1 if owner_model.layers[0].name == 'input_layer' else 0
# #             for lyr in owner_model.layers[start:]:
# #                 x = lyr(x)
# #                 if lyr.name == conv_name:
# #                     conv_out_tensor = x
# #             if conv_out_tensor is None:
# #                 print(f"ERROR: could not capture '{conv_name}'")
# #                 return None
# #             extractor = tf.keras.models.Model(
# #                 inputs=inp, outputs=[conv_out_tensor, x],
# #                 name="gradcam_extractor_seq"
# #             )

# #         if owner_model is model:
# #             run_extractor  = lambda t: extractor(t, training=False)
# #             run_full_model = lambda t: model(t, training=False)
# #         else:
# #             backbone_layer = next(
# #                 (l for l in model.layers
# #                  if l is owner_model or (
# #                      hasattr(l, 'layers') and
# #                      any(cl.name == conv_name for cl, _ in _collect_conv_layers(l))
# #                  )), None
# #             )
# #             if backbone_layer is None:
# #                 print("ERROR: backbone layer not found in top model")
# #                 return None
# #             try:
# #                 prefix = tf.keras.models.Model(
# #                     inputs=model.inputs, outputs=backbone_layer.input,
# #                     name="prefix_model"
# #                 )
# #             except Exception as e:
# #                 print(f"Prefix model failed: {e}")
# #                 return None

# #             run_extractor  = lambda t: extractor(prefix(t, training=False), training=False)
# #             run_full_model = lambda t: model(t, training=False)

# #         # Gradient tape
# #         with tf.GradientTape() as tape:
# #             conv_outputs, _ = run_extractor(img_tensor)
# #             tape.watch(conv_outputs)
# #             preds     = run_full_model(img_tensor)
# #             class_idx = int(tf.argmax(preds[0]).numpy())
# #             print(f"GradCAM class: {class_idx}  probs: {preds[0].numpy()}")
# #             score = tf.math.log(preds[:, class_idx] + 1e-8)

# #         grads = tape.gradient(score, conv_outputs)

# #         if grads is None:
# #             print("Grads None — single-graph fallback...")
# #             with tf.GradientTape() as tape2:
# #                 conv_outputs, owner_out = run_extractor(img_tensor)
# #                 tape2.watch(conv_outputs)
# #                 preds2     = owner_out
# #                 class_idx2 = int(tf.argmax(preds2[0]).numpy())
# #                 score2     = tf.math.log(preds2[:, class_idx2] + 1e-8)
# #             grads = tape2.gradient(score2, conv_outputs)
# #             if grads is None:
# #                 print("ERROR: fallback gradients also None")
# #                 return None

# #         pooled   = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
# #         conv_np  = conv_outputs[0].numpy()
# #         heatmap  = np.maximum(np.einsum('ijk,k->ij', conv_np, pooled), 0)

# #         hmax = np.max(heatmap)
# #         if hmax > 1e-8:
# #             heatmap /= hmax
# #         else:
# #             print("WARNING: ReLU heatmap all-zero, trying abs fallback...")
# #             heatmap = np.einsum('ijk,k->ij', conv_np, np.abs(pooled))
# #             hmax    = np.max(heatmap)
# #             if hmax > 1e-8:
# #                 heatmap /= hmax
# #             else:
# #                 print("ERROR: heatmap zero even with abs fallback")
# #                 return None

# #         return cv2.resize(heatmap, target_size)

# #     except Exception as e:
# #         print(f"Error in generate_gradcam_simple: {e}")
# #         import traceback; traceback.print_exc()
# #         return None


# # # ---------------------------------------------------------------------------
# # # Routes
# # # ---------------------------------------------------------------------------

# # @app.route('/')
# # def home():
# #     return render_template('index.html')


# # @app.route('/preview', methods=['GET', 'POST'])
# # def preview():
# #     if request.method == 'POST':
# #         file = request.files.get('image')
# #         if file and file.filename != "":
# #             filename = file.filename
# #             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #             file.save(filepath)
# #             return render_template('preview.html', filename=filename)
# #     return render_template('preview.html')


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         print("\n========== PREDICT ROUTE STARTED ==========")
# #         global cnn_model, tf, keras

# #         # Lazy-load TensorFlow
# #         if tf is None:
# #             import tensorflow as _tf
# #             tf = _tf
# #             try:
# #                 import keras as _keras
# #             except ImportError:
# #                 from tensorflow import keras as _keras
# #             keras = _keras
# #             print(f"TF {tf.__version__}  Keras {keras.__version__}")

# #         # Lazy-load CNN model
# #         if cnn_model is None:
# #             try:
# #                 print("Loading CNN model...")
# #                 cnn_model = keras.models.load_model(
# #                     'models/keras_cnn_model.h5', compile=False
# #                 )
# #                 _dummy = np.zeros((1, 128, 128, 3), dtype='float32')
# #                 cnn_model(_dummy, training=False)
# #                 n_classes = cnn_model.output_shape[-1]
# #                 print(f"✓ CNN loaded — input={cnn_model.input_shape} "
# #                       f"output={cnn_model.output_shape} ({n_classes} classes)")

# #                 # Safety check: warn if class count doesn't match mapping
# #                 if n_classes != len(CNN_CLASS_NAMES):
# #                     print(
# #                         f"⚠️  WARNING: model has {n_classes} outputs but "
# #                         f"CNN_CLASS_NAMES has {len(CNN_CLASS_NAMES)} entries. "
# #                         f"Run debug_classes.py to fix the mapping."
# #                     )
# #             except Exception as e:
# #                 print(f"✗ Error loading CNN model: {e}")
# #                 import traceback; traceback.print_exc()
# #                 return "Error loading CNN model", 500

# #         # Resolve uploaded file
# #         filename = request.form.get('image_name') or request.form.get('filename')
# #         print(f"1. Filename: {filename}")
# #         if not filename:
# #             return redirect(url_for('preview'))

# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         print(f"2. Path: {filepath}")
# #         if not os.path.exists(filepath):
# #             print(f"ERROR: file missing at {filepath}")
# #             return redirect(url_for('preview'))

# #         # Preprocess
# #         print("3. Preprocessing...")
# #         try:
# #             img_array = preprocess_for_cnn(filepath)
# #         except Exception as e:
# #             print(f"ERROR preprocessing: {e}")
# #             return redirect(url_for('preview'))

# #         if img_array is None:
# #             return redirect(url_for('preview'))
# #         print(f"   shape={img_array.shape}  range=[{img_array.min():.2f}, {img_array.max():.2f}]")

# #         # Predict
# #         print("4. Predicting...")
# #         class_idx, confidence = get_prediction(img_array)
# #         if class_idx is None:
# #             return redirect(url_for('preview'))
# #         print(f"   class={class_idx}  confidence={confidence:.2f}%")

# #         # GradCAM
# #         print("5. GradCAM...")
# #         heatmap = generate_gradcam_simple(cnn_model, img_array)

# #         gradcam_filename = None
# #         if heatmap is not None:
# #             try:
# #                 original_img = cv2.imread(filepath)
# #                 if original_img is not None:
# #                     original_img      = cv2.resize(original_img, GRADCAM_OUTPUT_SIZE)
# #                     heatmap_u8      = (heatmap * 255).astype(np.uint8)
# #                     # Invert u8 so high activation→low value.
# #                     # COLORMAP_JET: 0→red, 128→green, 255→blue
# #                     # After inversion: high activation→red, low→blue  ✓
# #                     heatmap_colored = cv2.applyColorMap(255 - heatmap_u8, cv2.COLORMAP_JET)
# #                     overlay         = cv2.addWeighted(original_img, 0.55, heatmap_colored, 0.45, 0)
# #                     base_name         = os.path.splitext(filename)[0]
# #                     gradcam_filename  = f"{base_name}_gradcam.png"
# #                     gradcam_path      = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
# #                     os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
# #                     if cv2.imwrite(gradcam_path, overlay):
# #                         print(f"   ✓ GradCAM saved → {gradcam_path}")
# #                     else:
# #                         print("   ERROR: imwrite failed")
# #                         gradcam_filename = None
# #             except Exception as e:
# #                 print(f"ERROR creating overlay: {e}")
# #                 gradcam_filename = None

# #         # Store & redirect
# #         prediction = CNN_CLASS_NAMES.get(class_idx, f"Unknown (idx {class_idx})")
# #         results = {
# #             'filename':         filename,
# #             'gradcam_filename': gradcam_filename,
# #             'prediction':       prediction,
# #             'confidence':       round(confidence, 2),
# #             'status':           STATUS_MESSAGES.get(class_idx, ""),
# #         }
# #         print(f"7. Results: {results}")
# #         session['results'] = results
# #         print("========== PREDICT DONE ==========")
# #         return redirect(url_for('show_results'))

# #     except Exception as e:
# #         print(f"ERROR in predict: {e}")
# #         import traceback; traceback.print_exc()
# #         return redirect(url_for('preview'))


# # @app.route('/results')
# # def show_results():
# #     results = session.get('results')
# #     if not results:
# #         return redirect(url_for('home'))
# #     # Pass each value as a direct template variable to match
# #     # portfolio-details.html which uses {{ prediction }}, {{ filename }}, etc.
# #     return render_template(
# #         'portfolio-details.html',
# #         filename         = results['filename'],
# #         gradcam_filename = results['gradcam_filename'],
# #         prediction       = results['prediction'],
# #         confidence       = results['confidence'],
# #         status           = results['status'],
# #     )


# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for, session
# import os
# import numpy as np
# from PIL import Image
# import cv2

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'

# # ---------------------------------------------------------------------------
# # Folder configuration
# # ---------------------------------------------------------------------------
# UPLOAD_FOLDER = 'static/uploads'
# GRADCAM_FOLDER = 'static/gradcam'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# GRADCAM_OUTPUT_SIZE = (512, 512)

# # ---------------------------------------------------------------------------
# # FIX 1 — Unknown / non-rice leaf detection
# # If the model's top confidence is below this threshold the image is treated
# # as an unknown / non-rice-leaf image.  No GradCAM is generated.
# # ---------------------------------------------------------------------------
# UNKNOWN_CONFIDENCE_THRESHOLD = 50.0   # percent

# # ---------------------------------------------------------------------------
# # Class label mappings
# # Keras ImageDataGenerator.flow_from_directory() assigns indices by sorting
# # training folder names ALPHABETICALLY.
# #   bacterial_leaf_blight → 0
# #   brown_spot            → 1
# #   healthy_rice_leaf     → 2
# #   leaf_blast            → 3
# # ---------------------------------------------------------------------------
# CNN_CLASS_NAMES = {
#     0: "Bacterial Leaf Blight",
#     1: "Brown Spot",
#     2: "Healthy Rice Leaf",
#     3: "Leaf Blast",
# }

# # Class index for Healthy — GradCAM is skipped for this class (FIX 2)
# HEALTHY_CLASS_IDX = 2

# STATUS_MESSAGES = {
#     -1: (
#         "The uploaded image does not appear to be a rice leaf, or the model "
#         "confidence is too low to make a reliable prediction. Please upload "
#         "a clear, close-up image of a rice leaf and try again."
#     ),
#     0: (
#         "Bacterial Leaf Blight detected. This is a serious bacterial disease. "
#         "Recommended actions: Remove and destroy affected leaves immediately, "
#         "apply copper-based bactericides (e.g. copper oxychloride), improve field "
#         "drainage, avoid overhead irrigation, and use resistant varieties."
#     ),
#     1: (
#         "Brown Spot detected. This fungal disease is often linked to nutrient "
#         "deficiency. Recommended actions: Remove infected leaves, apply fungicide "
#         "(e.g. mancozeb or propiconazole), improve soil nutrition especially "
#         "potassium and silicon, reduce plant density for better air circulation."
#     ),
#     2: (
#         "Healthy Rice Leaf detected. The plant appears healthy with no visible "
#         "signs of disease. Continue good agricultural practices: proper irrigation, "
#         "balanced NPK fertilisation, and regular field monitoring to catch any "
#         "early disease signs."
#     ),
#     3: (
#         "Leaf Blast detected. This is one of the most destructive rice diseases. "
#         "Recommended actions: Remove and destroy infected plant parts, apply "
#         "systemic fungicide (e.g. tricyclazole or isoprothiolane), avoid excessive "
#         "nitrogen fertilisation, improve air circulation, and consider resistant "
#         "varieties for future planting."
#     ),
# }

# # ---------------------------------------------------------------------------
# # Lazy-loaded TensorFlow globals
# # ---------------------------------------------------------------------------
# tf        = None
# keras     = None
# cnn_model = None


# # ---------------------------------------------------------------------------
# # Preprocessing  (must match training exactly)
# # ---------------------------------------------------------------------------
# def preprocess_for_cnn(image_path, target_size=(128, 128)):
#     """
#     Replicates Keras ImageDataGenerator(rescale=1/255) + flow_from_directory:
#       - PIL RGB open  (no BGR swap needed)
#       - LANCZOS resize  (matches Keras/PIL default)
#       - divide by 255
#     """
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize(target_size, Image.LANCZOS)
#     arr = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)   # (1, H, W, 3)


# # ---------------------------------------------------------------------------
# # CNN prediction
# # ---------------------------------------------------------------------------
# def get_prediction(image_array):
#     """Return (class_idx, confidence_percent) or (None, None) on error."""
#     try:
#         if cnn_model is None:
#             print("ERROR: CNN model not loaded")
#             return None, None

#         img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#         output     = cnn_model(img_tensor, training=False).numpy()
#         print(f"[cnn] raw output: {output[0]}")

#         if output.shape[1] > 1:                    # softmax (4-class)
#             class_idx  = int(np.argmax(output[0]))
#             confidence = float(np.max(output[0])) * 100
#         else:                                       # sigmoid (binary)
#             prob       = float(output[0][0])
#             class_idx  = 1 if prob >= 0.5 else 0
#             confidence = max(prob, 1.0 - prob) * 100

#         label = CNN_CLASS_NAMES.get(class_idx, "Unknown")
#         print(f"[cnn] class_idx={class_idx}  label='{label}'  confidence={confidence:.2f}%")
#         return class_idx, confidence

#     except Exception as e:
#         print(f"Error in get_prediction: {e}")
#         import traceback; traceback.print_exc()
#         return None, None


# # ---------------------------------------------------------------------------
# # GradCAM helpers
# # ---------------------------------------------------------------------------
# def _collect_conv_layers(model, _seen=None):
#     """Recursively collect all Conv2D layers, including nested sub-models."""
#     if _seen is None:
#         _seen = set()
#     results = []
#     for layer in model.layers:
#         if id(layer) in _seen:
#             continue
#         _seen.add(id(layer))
#         if hasattr(layer, 'layers'):
#             results.extend(_collect_conv_layers(layer, _seen))
#         elif type(layer).__name__ == 'Conv2D':
#             results.append((layer, model))
#     return results


# def _find_gradcam_conv_layer(model):
#     """
#     Return (owner_model, layer_name) for the last Conv2D that appears before
#     any GlobalPooling / Flatten layer — the standard GradCAM target.
#     """
#     COLLAPSES = {'GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'Flatten'}

#     all_conv = _collect_conv_layers(model)
#     if not all_conv:
#         return None

#     def _flat(m, seen=None):
#         if seen is None:
#             seen = set()
#         out = []
#         for l in m.layers:
#             if id(l) in seen:
#                 continue
#             seen.add(id(l))
#             if hasattr(l, 'layers'):
#                 out.extend(_flat(l, seen))
#             else:
#                 out.append(l)
#         return out

#     flat         = _flat(model)
#     collapse_idx = next(
#         (i for i, l in enumerate(flat) if type(l).__name__ in COLLAPSES), None
#     )

#     if collapse_idx is None:
#         layer, owner = all_conv[-1][0], all_conv[-1][1]
#         print(f"GradCAM: no collapse layer found, using last conv '{layer.name}'")
#         return owner, layer.name

#     best = None
#     for layer, owner in all_conv:
#         try:
#             idx = flat.index(layer)
#         except ValueError:
#             continue
#         if idx < collapse_idx:
#             best = (owner, layer.name)
#         else:
#             break

#     if best is None:
#         best = (all_conv[-1][1], all_conv[-1][0].name)

#     print(f"GradCAM: target conv='{best[1]}' in owner='{best[0].name}'")
#     return best


# def generate_gradcam_simple(model, image_array, target_class_idx,
#                              target_size=GRADCAM_OUTPUT_SIZE):
#     """
#     Correct GradCAM implementation.

#     FIX 3 — The previous version called run_extractor() and run_full_model()
#     as two SEPARATE forward passes inside the GradientTape.  Because the conv
#     outputs and the prediction scores came from different computation graphs,
#     tape.gradient(score, conv_outputs) always returned None or garbage — the
#     tensors were not connected.

#     The correct approach is ONE single forward pass inside the tape using a
#     single Model that outputs BOTH the conv feature maps AND the final
#     predictions at the same time.  That way conv_outputs and preds live in
#     the same graph and the gradient flows correctly.

#     We also pass target_class_idx explicitly so the gradient is always taken
#     w.r.t. the predicted class (not recomputed inside the tape, which could
#     pick a different class if argmax is non-differentiable).
#     """
#     try:
#         print("\n=== GRADCAM ===")
#         if image_array is None:
#             return None

#         img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#         if len(img_tensor.shape) == 3:
#             img_tensor = tf.expand_dims(img_tensor, axis=0)

#         # ── Find the best Conv2D layer ───────────────────────────────────────
#         result = _find_gradcam_conv_layer(model)
#         if result is None:
#             print("ERROR: no Conv2D layer found")
#             return None
#         owner_model, conv_name = result

#         # ── Build a single extractor model: input → [conv_output, final_pred] ─
#         # This is the KEY fix: both outputs come from the SAME forward pass so
#         # gradients flow from the prediction score back to the conv feature map.
#         try:
#             conv_layer = owner_model.get_layer(conv_name)
#             if owner_model is model:
#                 # Simple case: the conv layer belongs to the top-level model
#                 grad_model = tf.keras.models.Model(
#                     inputs  = model.inputs,
#                     outputs = [conv_layer.output, model.output],
#                     name    = "gradcam_single_pass"
#                 )
#                 get_inputs = lambda t: t
#             else:
#                 # Nested backbone: build prefix → extractor chain
#                 backbone_layer = next(
#                     (l for l in model.layers
#                      if l is owner_model or (
#                          hasattr(l, 'layers') and
#                          any(cl.name == conv_name
#                              for cl, _ in _collect_conv_layers(l))
#                      )), None
#                 )
#                 if backbone_layer is None:
#                     print("ERROR: backbone layer not found in top model")
#                     return None

#                 # Build: top_model.input → [conv_output, top_model.output]
#                 # by constructing a Model that threads through the backbone
#                 inp = model.inputs[0]
#                 x   = inp
#                 conv_out_tensor = None
#                 for layer in model.layers[1:]:          # skip InputLayer
#                     x = layer(x)
#                     # capture the output of the target conv layer wherever it is
#                     if hasattr(layer, 'layers'):
#                         # It's a sub-model — walk its layers to find conv_name
#                         sub_inp = layer.input
#                         sub_x   = sub_inp
#                         for sub_layer in layer.layers[1:]:
#                             sub_x = sub_layer(sub_x)
#                             if sub_layer.name == conv_name:
#                                 conv_out_tensor = sub_x
#                     elif layer.name == conv_name:
#                         conv_out_tensor = x

#                 if conv_out_tensor is None:
#                     print(f"ERROR: could not locate '{conv_name}' output in graph")
#                     return None

#                 grad_model = tf.keras.models.Model(
#                     inputs  = inp,
#                     outputs = [conv_out_tensor, x],
#                     name    = "gradcam_single_pass_nested"
#                 )
#                 get_inputs = lambda t: t

#         except Exception as e:
#             print(f"Model construction failed ({e}), trying Sequential chain fallback...")
#             # Sequential fallback: manually chain layers
#             inp = tf.keras.Input(shape=model.input_shape[1:])
#             x, conv_out_tensor = inp, None
#             start = 1 if model.layers[0].name == 'input_layer' else 0
#             for lyr in model.layers[start:]:
#                 x = lyr(x)
#                 if lyr.name == conv_name:
#                     conv_out_tensor = x
#             if conv_out_tensor is None:
#                 print(f"ERROR: '{conv_name}' not found during Sequential chaining")
#                 return None
#             grad_model = tf.keras.models.Model(
#                 inputs  = inp,
#                 outputs = [conv_out_tensor, x],
#                 name    = "gradcam_seq_fallback"
#             )
#             get_inputs = lambda t: t

#         # ── Single forward pass inside GradientTape ──────────────────────────
#         # Both conv_outputs and preds come from the SAME graph → gradients work
#         with tf.GradientTape() as tape:
#             conv_outputs, preds = grad_model(get_inputs(img_tensor), training=False)
#             tape.watch(conv_outputs)
#             # Use the externally-determined target class so argmax is not
#             # recomputed inside the tape (argmax is not differentiable)
#             score = preds[:, target_class_idx]
#             print(f"GradCAM target class: {target_class_idx}  "
#                   f"score: {score.numpy()[0]:.4f}  "
#                   f"all probs: {preds[0].numpy()}")

#         grads = tape.gradient(score, conv_outputs)

#         if grads is None:
#             print("ERROR: gradients are None even with single-pass model")
#             return None

#         print(f"   grad shape={grads.shape}  "
#               f"grad range=[{grads.numpy().min():.4f}, {grads.numpy().max():.4f}]")

#         # ── Compute weighted heatmap ─────────────────────────────────────────
#         # Global-average-pool the gradients over the spatial dimensions
#         pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()   # (C,)
#         conv_np = conv_outputs[0].numpy()                          # (H, W, C)

#         # Weighted sum of feature maps, then ReLU
#         heatmap = np.maximum(np.einsum('hwc,c->hw', conv_np, pooled), 0)

#         hmax = np.max(heatmap)
#         if hmax > 1e-8:
#             heatmap /= hmax
#         else:
#             print("WARNING: ReLU heatmap all-zero, falling back to abs(gradients)...")
#             heatmap = np.einsum('hwc,c->hw', conv_np, np.abs(pooled))
#             hmax    = np.max(heatmap)
#             if hmax > 1e-8:
#                 heatmap /= hmax
#             else:
#                 print("ERROR: heatmap zero even with abs fallback — returning None")
#                 return None

#         print(f"   heatmap range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")
#         return cv2.resize(heatmap, target_size)

#     except Exception as e:
#         print(f"Error in generate_gradcam_simple: {e}")
#         import traceback; traceback.print_exc()
#         return None


# # ---------------------------------------------------------------------------
# # Routes
# # ---------------------------------------------------------------------------

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/preview', methods=['GET', 'POST'])
# def preview():
#     if request.method == 'POST':
#         file = request.files.get('image')
#         if file and file.filename != "":
#             filename = file.filename
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             return render_template('preview.html', filename=filename)
#     return render_template('preview.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         print("\n========== PREDICT ROUTE STARTED ==========")
#         global cnn_model, tf, keras

#         # Lazy-load TensorFlow
#         if tf is None:
#             import tensorflow as _tf
#             tf = _tf
#             try:
#                 import keras as _keras
#             except ImportError:
#                 from tensorflow import keras as _keras
#             keras = _keras
#             print(f"TF {tf.__version__}  Keras {keras.__version__}")

#         # Lazy-load CNN model
#         if cnn_model is None:
#             try:
#                 print("Loading CNN model...")
#                 cnn_model = keras.models.load_model(
#                     'models/keras_cnn_model.h5', compile=False
#                 )
#                 _dummy = np.zeros((1, 128, 128, 3), dtype='float32')
#                 cnn_model(_dummy, training=False)
#                 n_classes = cnn_model.output_shape[-1]
#                 print(f"✓ CNN loaded — input={cnn_model.input_shape} "
#                       f"output={cnn_model.output_shape} ({n_classes} classes)")
#                 if n_classes != len(CNN_CLASS_NAMES):
#                     print(f"⚠️  WARNING: model has {n_classes} outputs but "
#                           f"CNN_CLASS_NAMES has {len(CNN_CLASS_NAMES)} entries.")
#             except Exception as e:
#                 print(f"✗ Error loading CNN model: {e}")
#                 import traceback; traceback.print_exc()
#                 return "Error loading CNN model", 500

#         # Resolve uploaded file
#         filename = request.form.get('image_name') or request.form.get('filename')
#         print(f"1. Filename: {filename}")
#         if not filename:
#             return redirect(url_for('preview'))

#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         print(f"2. Path: {filepath}")
#         if not os.path.exists(filepath):
#             print(f"ERROR: file missing at {filepath}")
#             return redirect(url_for('preview'))

#         # Preprocess
#         print("3. Preprocessing...")
#         try:
#             img_array = preprocess_for_cnn(filepath)
#         except Exception as e:
#             print(f"ERROR preprocessing: {e}")
#             return redirect(url_for('preview'))

#         if img_array is None:
#             return redirect(url_for('preview'))
#         print(f"   shape={img_array.shape}  "
#               f"range=[{img_array.min():.2f}, {img_array.max():.2f}]")

#         # Predict
#         print("4. Predicting...")
#         class_idx, confidence = get_prediction(img_array)
#         if class_idx is None:
#             return redirect(url_for('preview'))
#         print(f"   class={class_idx}  confidence={confidence:.2f}%")

#         # ── FIX 1: Unknown / non-rice leaf detection ─────────────────────────
#         # If confidence is below threshold, treat as unknown regardless of
#         # which class the model picked.
#         is_unknown = confidence < UNKNOWN_CONFIDENCE_THRESHOLD
#         if is_unknown:
#             print(f"   ⚠ Confidence {confidence:.2f}% < threshold "
#                   f"{UNKNOWN_CONFIDENCE_THRESHOLD}% → treating as UNKNOWN")

#         # ── FIX 2: Skip GradCAM for Healthy and Unknown ──────────────────────
#         # GradCAM is only meaningful for disease classes.
#         # - Unknown images: no GradCAM (not a rice leaf)
#         # - Healthy rice leaf (class 2): no GradCAM needed
#         # - Disease classes (0, 1, 3): generate GradCAM
#         need_gradcam = (not is_unknown) and (class_idx != HEALTHY_CLASS_IDX)

#         gradcam_filename = None
#         if need_gradcam:
#             print("5. Generating GradCAM (disease class detected)...")
#             # ── FIX 3: Pass target_class_idx so the single-pass model uses
#             #           the correct class score for gradient computation
#             heatmap = generate_gradcam_simple(cnn_model, img_array,
#                                               target_class_idx=class_idx)
#             if heatmap is not None:
#                 try:
#                     original_img = cv2.imread(filepath)
#                     if original_img is not None:
#                         original_img = cv2.resize(original_img, GRADCAM_OUTPUT_SIZE)

#                         heatmap_u8 = (heatmap * 255).astype(np.uint8)

#                         # COLORMAP_JET: value 0 → red, 128 → green, 255 → blue
#                         # Invert so that HIGH activation (diseased) → RED  ✓
#                         #              LOW  activation (healthy)   → BLUE ✓
#                         heatmap_colored = cv2.applyColorMap(
#                             255 - heatmap_u8, cv2.COLORMAP_JET
#                         )
#                         overlay = cv2.addWeighted(
#                             original_img, 0.55, heatmap_colored, 0.45, 0
#                         )

#                         base_name        = os.path.splitext(filename)[0]
#                         gradcam_filename = f"{base_name}_gradcam.png"
#                         gradcam_path     = os.path.join(
#                             app.config['GRADCAM_FOLDER'], gradcam_filename
#                         )
#                         os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
#                         if cv2.imwrite(gradcam_path, overlay):
#                             print(f"   ✓ GradCAM saved → {gradcam_path}")
#                         else:
#                             print("   ERROR: imwrite failed")
#                             gradcam_filename = None
#                 except Exception as e:
#                     print(f"ERROR creating overlay: {e}")
#                     gradcam_filename = None
#             else:
#                 print("   GradCAM returned None — skipping overlay")
#         else:
#             reason = "unknown image" if is_unknown else "healthy leaf"
#             print(f"5. Skipping GradCAM ({reason})")

#         # ── Build result dict ─────────────────────────────────────────────────
#         if is_unknown:
#             prediction = "Unknown / Non-Rice Leaf"
#             status     = STATUS_MESSAGES[-1]
#         else:
#             prediction = CNN_CLASS_NAMES.get(class_idx, f"Unknown (idx {class_idx})")
#             status     = STATUS_MESSAGES.get(class_idx, "")

#         results = {
#             'filename':         filename,
#             'gradcam_filename': gradcam_filename,   # None for healthy/unknown
#             'prediction':       prediction,
#             'confidence':       round(confidence, 2),
#             'status':           status,
#             'is_unknown':       is_unknown,
#         }
#         print(f"6. Results: {results}")
#         session['results'] = results
#         print("========== PREDICT DONE ==========")
#         return redirect(url_for('show_results'))

#     except Exception as e:
#         print(f"ERROR in predict: {e}")
#         import traceback; traceback.print_exc()
#         return redirect(url_for('preview'))


# @app.route('/results')
# def show_results():
#     results = session.get('results')
#     if not results:
#         return redirect(url_for('home'))
#     return render_template(
#         'portfolio-details.html',
#         filename         = results['filename'],
#         gradcam_filename = results['gradcam_filename'],
#         prediction       = results['prediction'],
#         confidence       = results['confidence'],
#         status           = results['status'],
#         is_unknown       = results.get('is_unknown', False),
#     )


# if __name__ == '__main__':
#     app.run(debug=True)

# -------------------- IMPORTS --------------------

from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ---------------------------------------------------------------------------
# Folder configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

DISPLAY_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Class label mappings
# ---------------------------------------------------------------------------
CNN_CLASS_NAMES = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy Rice Leaf",
    3: "Leaf Blast",
}

HEALTHY_CLASS_IDX = 2

STATUS_MESSAGES = {
    0: (
        "Bacterial Leaf Blight detected. This is a serious bacterial disease. "
        "Recommended actions: Remove and destroy affected leaves immediately, "
        "apply copper-based bactericides (e.g. copper oxychloride), improve field "
        "drainage, avoid overhead irrigation, and use resistant varieties."
    ),
    1: (
        "Brown Spot detected. This fungal disease is often linked to nutrient "
        "deficiency. Recommended actions: Remove infected leaves, apply fungicide "
        "(e.g. mancozeb or propiconazole), improve soil nutrition especially "
        "potassium and silicon, reduce plant density for better air circulation."
    ),
    2: (
        "Healthy Rice Leaf detected. The plant appears healthy with no visible "
        "signs of disease. Continue good agricultural practices: proper irrigation, "
        "balanced NPK fertilisation, and regular field monitoring to catch any "
        "early disease signs."
    ),
    3: (
        "Leaf Blast detected. This is one of the most destructive rice diseases. "
        "Recommended actions: Remove and destroy infected plant parts, apply "
        "systemic fungicide (e.g. tricyclazole or isoprothiolane), avoid excessive "
        "nitrogen fertilisation, improve air circulation, and consider resistant "
        "varieties for future planting."
    ),
}

# ---------------------------------------------------------------------------
# Lazy-loaded TensorFlow globals
# ---------------------------------------------------------------------------
tf        = None
keras     = None
cnn_model = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_cnn(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# CNN prediction
# ---------------------------------------------------------------------------
def get_prediction(image_array):
    try:
        if cnn_model is None:
            print("ERROR: CNN model not loaded")
            return None, None, None

        img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        output     = cnn_model(img_tensor, training=False).numpy()
        print(f"[cnn] raw output: {output[0]}")

        if output.shape[1] > 1:
            probs      = output[0]
            class_idx  = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100
        else:
            prob       = float(output[0][0])
            class_idx  = 1 if prob >= 0.5 else 0
            confidence = max(prob, 1.0 - prob) * 100
            probs      = np.array([1.0 - prob, prob])

        label = CNN_CLASS_NAMES.get(class_idx, "Unknown")
        print(f"[cnn] class_idx={class_idx}  label='{label}'  confidence={confidence:.2f}%")
        return class_idx, confidence, probs

    except Exception as e:
        print(f"Error in get_prediction: {e}")
        import traceback; traceback.print_exc()
        return None, None, None


# ===========================================================================
# LEAF BODY MASK
# ===========================================================================
def get_leaf_body_mask(image_bgr, target_size):
    resized = cv2.resize(image_bgr, target_size)
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    plant_mask = (
        (H >= 10) & (H <= 90) &
        (S >  30) &
        (V >  35)
    ).astype(np.uint8) * 255

    lesion_mask = (
        (H >= 5) & (H <= 20) &
        (S > 40) &
        (V > 25)
    ).astype(np.uint8) * 255

    combined = cv2.bitwise_or(plant_mask, lesion_mask)

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=3)

    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open,  iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    kept = np.zeros_like(combined)
    if num_labels > 1:
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
        areas.sort(reverse=True)
        for _, lbl in areas[:3]:
            kept[labels == lbl] = 255

    mask_float = cv2.GaussianBlur(kept.astype(np.float32), (31, 31), 0) / 255.0
    coverage   = mask_float.mean()
    print(f"[leaf-mask] coverage={coverage:.2%}")

    if coverage < 0.08:
        print("  → mask too small, using full-image fallback")
        return np.ones((target_size[1], target_size[0]), dtype=np.float32)

    return mask_float


# ===========================================================================
# DISEASE PRESENCE MAP
# ===========================================================================
def get_disease_presence_map(image_bgr, target_size, leaf_mask):
    resized = cv2.resize(image_bgr, target_size)
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    green_hue     = np.clip(1.0 - np.abs(H - 60.0) / 30.0, 0, 1)
    green_sat     = np.clip((S - 60.0) / 120.0, 0, 1)
    green_val     = np.clip((V - 60.0) / 120.0, 0, 1)
    greenness     = green_hue * green_sat * green_val
    disease_score = 1.0 - greenness

    dark_boost    = np.clip(1.0 - V / 80.0,  0, 1)
    pale_boost    = np.clip(1.0 - S / 60.0,  0, 1) * np.clip(V / 150.0, 0, 1)
    disease_score = np.clip(disease_score + dark_boost * 0.5 + pale_boost * 0.5, 0, 1)

    disease_score = disease_score * leaf_mask

    disease_score = cv2.GaussianBlur(disease_score, (31, 31), 0)
    dmax = disease_score.max()
    if dmax > 1e-6:
        disease_score /= dmax

    return disease_score.astype(np.float32)


# ===========================================================================
# GRADCAM — per-layer computation
# ===========================================================================
def _collect_conv_layers(model, _seen=None):
    if _seen is None:
        _seen = set()
    results = []
    for layer in model.layers:
        if id(layer) in _seen:
            continue
        _seen.add(id(layer))
        if hasattr(layer, 'layers'):
            results.extend(_collect_conv_layers(layer, _seen))
        elif type(layer).__name__ == 'Conv2D':
            results.append((layer, model))
    return results


def _compute_heatmap_for_layer(model, img_tensor, target_class_idx,
                                owner_model, conv_name, target_size):
    try:
        conv_layer = owner_model.get_layer(conv_name)
        if owner_model is model:
            grad_model = tf.keras.models.Model(
                inputs  = model.inputs,
                outputs = [conv_layer.output, model.output],
                name    = f"gc_{conv_name}"
            )
        else:
            inp = model.inputs[0]
            x   = inp
            cot = None
            for layer in model.layers:
                if layer.name == 'input_layer' or hasattr(layer, '_is_input_layer'):
                    continue
                try:
                    x = layer(x)
                except Exception:
                    continue
                if hasattr(layer, 'layers'):
                    for sl in layer.layers:
                        if sl.name == conv_name:
                            cot = x; break
                elif layer.name == conv_name:
                    cot = x
            if cot is None:
                return None
            grad_model = tf.keras.models.Model(
                inputs=inp, outputs=[cot, x], name=f"gc_nested_{conv_name}"
            )
    except Exception:
        try:
            inp = tf.keras.Input(shape=model.input_shape[1:])
            x, cot = inp, None
            for lyr in model.layers:
                if 'input' in lyr.name:
                    continue
                x = lyr(x)
                if lyr.name == conv_name:
                    cot = x
            if cot is None:
                return None
            grad_model = tf.keras.models.Model(
                inputs=inp, outputs=[cot, x], name=f"gc_seq_{conv_name}"
            )
        except Exception:
            return None

    try:
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            score = tf.reduce_sum(preds[:, target_class_idx])

        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(img_tensor)
                conv_outputs, preds = grad_model(img_tensor, training=False)
                tape2.watch(conv_outputs)
                score = tf.reduce_sum(preds[:, target_class_idx])
            grads = tape2.gradient(score, conv_outputs)
            del tape2
            if grads is None:
                return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_np = conv_outputs[0].numpy()
        heatmap = np.maximum(np.einsum('hwc,c->hw', conv_np, pooled), 0)
        hmax    = heatmap.max()
        if hmax < 1e-8:
            heatmap = np.einsum('hwc,c->hw', conv_np, np.abs(pooled))
            hmax    = heatmap.max()
        if hmax < 1e-8:
            return None
        heatmap /= hmax
        return cv2.resize(heatmap, target_size)

    except Exception:
        return None


def generate_gradcam_best_layer(model, image_array, target_class_idx,
                                original_bgr, target_size=DISPLAY_SIZE):
    print("\n=== GRADCAM (multi-layer best-match with leaf masking) ===")
    if image_array is None:
        return None, None

    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    leaf_mask    = get_leaf_body_mask(original_bgr, target_size)
    disease_map  = get_disease_presence_map(original_bgr, target_size, leaf_mask)
    disease_flat = disease_map.flatten()
    disease_std  = disease_flat.std()
    print(f"   disease_map std={disease_std:.4f}")

    all_conv = _collect_conv_layers(model)
    if not all_conv:
        print("ERROR: no Conv2D layers found")
        return None, None
    print(f"   Found {len(all_conv)} Conv2D layers — evaluating all...")

    best_heatmap = None
    best_score   = -np.inf
    best_name    = None

    for conv_layer, owner_model in all_conv:
        conv_name = conv_layer.name
        heatmap   = _compute_heatmap_for_layer(
            model, img_tensor, target_class_idx,
            owner_model, conv_name, target_size
        )
        if heatmap is None:
            continue

        masked_h = heatmap * leaf_mask
        h_flat   = masked_h.flatten()
        h_std    = h_flat.std()

        score = 0.0
        if disease_std > 1e-6 and h_std > 1e-6:
            score = float(np.corrcoef(h_flat, disease_flat)[0, 1])

        print(f"   layer='{conv_name}'  correlation={score:.4f}")

        if score > best_score:
            best_score   = score
            best_heatmap = heatmap
            best_name    = conv_name

    if best_heatmap is None:
        print("ERROR: no valid heatmap from any layer")
        return None, None

    print(f"   ✓ Best layer='{best_name}'  score={best_score:.4f}")
    return best_heatmap, leaf_mask


# ===========================================================================
# GRADCAM OVERLAY
# ===========================================================================
def build_gradcam_overlay(original_bgr, heatmap, leaf_mask, display_size):
    original_resized = cv2.resize(original_bgr, display_size)

    masked_heatmap = heatmap * leaf_mask

    leaf_vals = masked_heatmap[leaf_mask > 0.5]
    if len(leaf_vals) == 0 or leaf_vals.max() < 1e-8:
        leaf_vals = masked_heatmap.flatten()

    p_low  = float(np.percentile(leaf_vals, 70))
    p_high = float(np.percentile(leaf_vals, 99))

    if p_high - p_low < 1e-6:
        stretched = masked_heatmap.copy()
    else:
        stretched = (masked_heatmap - p_low) / (p_high - p_low)
        stretched = np.clip(stretched, 0.0, 1.0)

    stretched = stretched * leaf_mask

    print(f"[overlay] p_low={p_low:.4f}  p_high={p_high:.4f}  "
          f"stretched=[{stretched.min():.3f}, {stretched.max():.3f}]")

    heatmap_u8      = (stretched * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    leaf_mask_3ch = np.stack([leaf_mask] * 3, axis=-1)
    overlay = (
        original_resized.astype(np.float32) * (1.0 - leaf_mask_3ch * 0.45) +
        heatmap_colored.astype(np.float32)  * (leaf_mask_3ch * 0.45)
    ).clip(0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/preview', methods=['GET', 'POST'])
def preview():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != "":
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('preview.html', filename=filename)
    return render_template('preview.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n========== PREDICT ROUTE STARTED ==========")
        global cnn_model, tf, keras

        if tf is None:
            import tensorflow as _tf
            tf = _tf
            try:
                import keras as _keras
            except ImportError:
                from tensorflow import keras as _keras
            keras = _keras
            print(f"TF {tf.__version__}  Keras {keras.__version__}")

        if cnn_model is None:
            try:
                print("Loading CNN model...")
                cnn_model = keras.models.load_model(
                    'models/keras_cnn_model.h5', compile=False
                )
                _dummy = np.zeros((1, 128, 128, 3), dtype='float32')
                cnn_model(_dummy, training=False)
                n_classes = cnn_model.output_shape[-1]
                print(f"✓ CNN loaded — input={cnn_model.input_shape} "
                      f"output={cnn_model.output_shape} ({n_classes} classes)")
                if n_classes != len(CNN_CLASS_NAMES):
                    print(f"⚠️  WARNING: model has {n_classes} outputs but "
                          f"CNN_CLASS_NAMES has {len(CNN_CLASS_NAMES)} entries.")
            except Exception as e:
                print(f"✗ Error loading CNN model: {e}")
                import traceback; traceback.print_exc()
                return "Error loading CNN model", 500

        filename = request.form.get('image_name') or request.form.get('filename')
        print(f"1. Filename: {filename}")
        if not filename:
            return redirect(url_for('preview'))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"2. Path: {filepath}")
        if not os.path.exists(filepath):
            print(f"ERROR: file missing at {filepath}")
            return redirect(url_for('preview'))

        base_name        = os.path.splitext(filename)[0]
        resized_filename = f"{base_name}_display.jpg"
        resized_path     = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
        original_img     = cv2.imread(filepath)

        try:
            if original_img is not None:
                cv2.imwrite(resized_path, cv2.resize(original_img, DISPLAY_SIZE))
                print(f"   ✓ Resized original saved → {resized_path}")
            else:
                resized_filename = filename
        except Exception as e:
            print(f"WARNING: could not save resized original: {e}")
            resized_filename = filename

        print("3. Preprocessing...")
        try:
            img_array = preprocess_for_cnn(filepath)
        except Exception as e:
            print(f"ERROR preprocessing: {e}")
            return redirect(url_for('preview'))

        if img_array is None:
            return redirect(url_for('preview'))

        print("4. Predicting...")
        class_idx, confidence, probs = get_prediction(img_array)
        if class_idx is None:
            return redirect(url_for('preview'))
        print(f"   class={class_idx}  confidence={confidence:.2f}%")

        need_gradcam     = (class_idx != HEALTHY_CLASS_IDX)
        gradcam_filename = None

        if need_gradcam:
            print("5. Generating GradCAM...")
            heatmap, leaf_mask = generate_gradcam_best_layer(
                cnn_model, img_array,
                target_class_idx=class_idx,
                original_bgr=original_img,
                target_size=DISPLAY_SIZE
            )
            if heatmap is not None and leaf_mask is not None:
                try:
                    overlay = build_gradcam_overlay(
                        original_img, heatmap, leaf_mask, DISPLAY_SIZE
                    )
                    gradcam_filename = f"{base_name}_gradcam.png"
                    gradcam_path     = os.path.join(
                        app.config['GRADCAM_FOLDER'], gradcam_filename
                    )
                    os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
                    if cv2.imwrite(gradcam_path, overlay):
                        print(f"   ✓ GradCAM saved → {gradcam_path}")
                    else:
                        print("   ERROR: imwrite failed")
                        gradcam_filename = None
                except Exception as e:
                    print(f"ERROR creating overlay: {e}")
                    import traceback; traceback.print_exc()
                    gradcam_filename = None
            else:
                print("   GradCAM returned None — skipping overlay")
        else:
            print("5. Skipping GradCAM — healthy leaf, no disease to highlight")

        prediction = CNN_CLASS_NAMES.get(class_idx, f"Unknown (idx {class_idx})")
        status     = STATUS_MESSAGES.get(class_idx, "")

        results = {
            'filename':         resized_filename,
            'gradcam_filename': gradcam_filename,
            'prediction':       prediction,
            'confidence':       round(confidence, 2),
            'status':           status,
            'is_unknown':       False,
        }
        print(f"6. Results: {results}")
        session['results'] = results
        print("========== PREDICT DONE ==========")
        return redirect(url_for('show_results'))

    except Exception as e:
        print(f"ERROR in predict: {e}")
        import traceback; traceback.print_exc()
        return redirect(url_for('preview'))


@app.route('/results')
def show_results():
    results = session.get('results')
    if not results:
        return redirect(url_for('home'))
    return render_template(
        'portfolio-details.html',
        filename         = results['filename'],
        gradcam_filename = results['gradcam_filename'],
        prediction       = results['prediction'],
        confidence       = results['confidence'],
        status           = results['status'],
        is_unknown       = results.get('is_unknown', False),
    )


# ── NEW ROUTE ──────────────────────────────────────────────────────────────
@app.route('/analyze-another')
def analyze_another():
    """Clear previous results from session and redirect to the home page."""
    session.pop('results', None)
    return redirect(url_for('home'))
# ──────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    app.run(debug=True)