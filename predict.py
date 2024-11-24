from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

CLASS_PATH = "classes.txt"

model = load_model('best_model.keras')
class_names = [line.strip() for line in open(CLASS_PATH)]

def preprocess_image(image):
    image = image.resize((224, 224)) # resize the image
    image = np.array(image) # convert to numpy array
    image = image / 255.0 # scale pixel values to 0-1
    image = np.expand_dims(image, axis=0) # add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image provided'}), 400
    image = request.files['image']
    try:
        image = Image.open(image)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        
        return jsonify({'dish_name': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)