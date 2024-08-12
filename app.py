from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

# Load the trained model
model_path = 'my_alexnet_model.h5'
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['Safe', 'Text', 'Talk', 'Other', 'Turn']

app = Flask(__name__)

def preprocess_image(img):
    """Preprocess the image for prediction."""
    img = img.resize((240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(img):
    """Predict the class of the image using the trained model."""
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))  # Convert to standard Python float
    return predicted_class, confidence

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        img = Image.open(image_bytes)
        predicted_class, confidence = predict_image(img)
        return jsonify(prediction=predicted_class, confidence=confidence)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
