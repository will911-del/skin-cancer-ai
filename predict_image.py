import tensorflow as tf
import numpy as np
import cv2
import sys

# Load the trained model
model = tf.keras.models.load_model("skin_cancer_model.h5")

# Class names (in the same order used during training)
class_names = ['nv', 'mel', 'bkl']

# Load image path from command line
if len(sys.argv) < 2:
    print("❗ Usage: python predict_image.py path_to_image.jpg")
    sys.exit()

image_path = sys.argv[1]

# Load and preprocess the image
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Error: Image not found at {image_path}")
    sys.exit()

img = cv2.resize(img, (64, 64))  # Match training size
img = img / 255.0                # Normalize
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img)
predicted_class = np.argmax(predictions)

# Output result
print(f"🔍 Predicted class: {class_names[predicted_class]}")

