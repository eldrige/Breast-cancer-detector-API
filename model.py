import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load Dataset (Replace with actual dataset)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./dataset/test/",
    image_size=(299, 299),
    batch_size=32,
    label_mode="binary"
)

# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(299, 299, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (cancer/no cancer)
])

# Compile & Train Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Save Model
model.save("breast_cancer_model.h5")
