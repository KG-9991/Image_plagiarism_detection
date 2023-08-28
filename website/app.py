from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import warnings
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
# Keras
import tensorflow as tf
#graph = tf.get_default_graph()
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template

#from gevent.pywsgi import WSGIServer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
model="new_model.hp5"


from flask_uploads import UploadSet, configure_uploads, IMAGES
# Define a flask app


app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploaded_images'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
from keras.preprocessing import image
def model_predict(file_path):
    #tf.keras.backend.clear_session()
    #graph = tf.get_default_graph()
    img = plt.imread(file_path) 
    image_size=180
    img = image.load_img(file_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
        #test_image = basemodel.predict(test_image)
    model12=tf.keras.models.load_model('new_model.hp5')
    print('hi2')
    predictions = model12.predict_classes(images)
    print('hi3')
    
    #predictions=[0]
    print(predictions)
    return predictions

@app.route('/index.html')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/paper.html',methods=["GET"])
def paper():
    return render_template('paper.html')
@app.route('/modifications.html')
def modifications():
    return render_template('modifications.html')
@app.route('/chatbot.html')
def chatbot():
    return render_template('chatbot.html')
@app.route('/predictions.html', methods=['GET', 'POST'])
def predictions():
    classes = {'TRAIN': ['GAN', 'Non-GAN'],
               'VALIDATION': ['GAN', 'Non-GAN'],
               'TEST': ['GAN', 'Non-GAN']}
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        pic = f.filename
        photo = pic.replace("'", "")
        picture = photo.replace(" ", "_")
        save_photo = photos.save(f)
        print('Uploaded')
        # Save the file to ./uploads
       
        file_path ="static/uploaded_images/"+pic
      
        # Make a prediction
        prediction = model_predict(file_path)
        predicted_class = classes['TRAIN'][prediction[0]]
        print('We think that is {}.'.format(predicted_class.lower()))
        return str(predicted_class).lower()
    return render_template('predictions.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    classes = {'TRAIN': ['GAN', 'Non-GAN'],
               'VALIDATION': ['GAN', 'Non-GAN'],
               'TEST': ['GAN', 'Non-GAN']}
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        pic = f.filename
        photo = pic.replace("'", "")
        picture = photo.replace(" ", "_")
        save_photo = photos.save(f)
        print('Uploaded')
        # Save the file to ./uploads
       
        file_path ="static/uploaded_images/"+pic
      
        # Make a prediction
        prediction = model_predict(file_path)
        predicted_class = classes['TRAIN'][prediction[0]]
        print('We think that is {}.'.format(predicted_class.lower()))
        return str(predicted_class).lower()
    return render_template('predictions.html')

if __name__ == '__main__':
    app.run('0.0.0.0')

