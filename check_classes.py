from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_PATH = "datasets"
datagen = ImageDataGenerator(rescale=1./255)

# We want to see how many classes it finds and what their indices are
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

print("Found classes:", train_data.class_indices)
