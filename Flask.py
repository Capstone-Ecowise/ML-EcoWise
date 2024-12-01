import numpy as np
import tensorflow as tf
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = load_model('./Model V2B2.h5')
class_names = ['battery', 'biological', 'cardboard', 'clothes','glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = tf.image.resize(image_array, (224, 224))
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    probability = predictions[0][class_index]

    return jsonify({
        'prediction': class_name,
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)