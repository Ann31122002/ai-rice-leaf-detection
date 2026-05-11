import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Paths and parameters
DATASET_DIR = 'dataset/Rice_Leaf_AUG'

IMG_SIZE = (128, 128)  # Reduced to 128x128 to save memory
BATCH_SIZE = 16  # Reduced to 16 to save memory
EPOCHS = 40  # Try more epochs for better learning

# Remove incorrect/corrupted images
def remove_incorrect_images(dataset_dir):
    removed = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print(f"Removing corrupted or non-image file: {file_path}")
                os.remove(file_path)
                removed += 1
    print(f"Removed {removed} incorrect/corrupted images.")

remove_incorrect_images(DATASET_DIR)

# Data generators (same as preprocessing)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Visualize sample images from the training set
import matplotlib.pyplot as plt
class_names = list(train_generator.class_indices.keys())
images, labels = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(class_names[labels[i].argmax()])
    plt.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.show()

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Model definition

# Deeper CNN with more filters and lower dropout
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Print final training and validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/keras_cnn_model.h5')

print("Model saved to models/keras_cnn_model.h5")

# Model evaluation: accuracy, confusion matrix, precision, recall
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Get true labels and predictions for validation set
val_steps = val_generator.samples // val_generator.batch_size + 1
val_generator.reset()
y_true = []
y_pred = []
for _ in range(val_steps):
    images, labels = next(val_generator)
    y_true.extend(np.argmax(labels, axis=1))
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys())

print(f"\nValidation Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
