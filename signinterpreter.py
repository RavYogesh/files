# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
train_dir = "asl_alphabet_train/asl_alphabet_train"
test_dir = "asl_alphabet_test/asl_alphabet_test"

# Define the image size and batch size
img_size = 224
batch_size = 32

# Create an image data generator with some data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # Use 20% of the data for validation
)

# Create the train and validation generators
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training" # Use 80% of the data for training
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation" # Use 20% of the data for validation
)

# Create a test generator
test_gen = ImageDataGenerator(rescale=1./255)

test_gen = test_gen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False # Do not shuffle the test data
)

# Get the number of classes and the class names
num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())

# Create a MobileNetV2 model with imagenet weights
base_model = keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the base model
base_model.trainable = False

# Add a global average pooling layer
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)

# Add a dropout layer
x = layers.Dropout(0.2)(x)

# Add a dense layer with softmax activation
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Create the final model
model = keras.Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Train the model
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Evaluate the model on the test data
model.evaluate(test_gen)

# Save the model
model.save("asl_model.h5")
