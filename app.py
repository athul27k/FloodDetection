import os
import sys
import flask

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)




from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')




# Model saved with Keras model.save()
MODEL_PATH = 'models/fine_tuned_flood_detection_model'


model = load_model(MODEL_PATH)
model.make_predict_function()          
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    img_array = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    
    preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    # preprocessed_image = model_predict(img, model)
    predictions = model.predict(preprocessed_image)
    return predictions


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')

@app.route('/services', methods=['GET'])
def index():
    # Service page
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    # Login page
    return render_template('login.html')  

@app.route('/signup', methods=['GET','POST'])
def sign_up():
    # Sign up page
    return render_template('sign_up.html')  



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.jpg")

        # Make prediction
        predictions = model_predict(img, model)

        # Process your result for human
        result = np.argmax(predictions)    # Max probability
        #pred_class = decode_predictions(predictions)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        #result = result.replace('_', ' ').capitalize()
        result="Flooding" if result==0 else "No Flooding"
        
        
        # Serialize the result, you can add additional fields
        return str(result)

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
