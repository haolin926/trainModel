import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


CLASS_PATH = "classes.txt"

model = load_model('best_model.keras')
class_names = [line.strip() for line in open(CLASS_PATH)]

def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    image = Image.open(image_path)  # Open the image
    image = image.resize((224, 224))  # Resize to match model's expected input
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Test on a single image
image_path = 'chicken_curry.jpg'  # Replace with an actual test image path
image = preprocess_image(image_path)

# Predict
predictions = model.predict(image)
predicted_class = class_names[np.argmax(predictions[0])]

print(f"Predicted Class: {predicted_class}")