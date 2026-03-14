"""
Converts my_chess_model.v2.keras → chess_model.tflite
Run this ONCE locally before deploying.
"""
import tensorflow as tf

KERAS_MODEL_PATH  = "my_chess_model.v2.keras"
TFLITE_MODEL_PATH = "chess_model.tflite"

print(f"[convert] Loading Keras model from '{KERAS_MODEL_PATH}'...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

print("[convert] Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization: quantize weights from float32 → int8
# This shrinks the model size by ~4x and reduces RAM usage significantly
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print(f"[convert] Saving TFLite model to '{TFLITE_MODEL_PATH}'...")
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

# Report size comparison
import os
keras_size  = os.path.getsize(KERAS_MODEL_PATH)  / (1024 * 1024)
tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)

print(f"\n[convert] Done!")
print(f"  Original Keras model : {keras_size:.1f} MB")
print(f"  TFLite model         : {tflite_size:.1f} MB")
print(f"  Size reduction       : {((keras_size - tflite_size) / keras_size * 100):.0f}%")
print(f"\nNext step: upload '{TFLITE_MODEL_PATH}' to Google Drive and update GDRIVE_FILE_ID on Render.")
