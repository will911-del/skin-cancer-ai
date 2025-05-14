import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load metadata
df = pd.read_csv('HAM10000_metadata.csv')

# Use only 3 classes to simplify
df = df[df['dx'].isin(['nv', 'mel', 'bkl'])]

# Label encode
label_map = {'nv': 0, 'mel': 1, 'bkl': 2}
df['label'] = df['dx'].map(label_map)

# Prepare data
images = []
labels = []

for i, row in df.iterrows():
    img_path = row['image_id'] + ".jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        labels.append(row['label'])

X = np.array(images) / 255.0
y = to_categorical(labels, num_classes=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Save the model
model.save("skin_cancer_model.h5")
print("✅ Model training complete and saved!")

