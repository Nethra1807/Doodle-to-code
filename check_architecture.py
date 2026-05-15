import tensorflow as tf

model_path = "shape_model.keras"
model = tf.keras.models.load_model(model_path)
print("Model Summary:")
model.summary()

output_shape = model.output_shape
print(f"Output shape: {output_shape}")

# Also check class indices if stored in the model metadata (common in newer Keras)
# But here we probably just need to know the number of neurons in the last layer.
