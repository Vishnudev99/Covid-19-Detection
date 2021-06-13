from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
import cv2
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
import os
model_file = "model.h5"
model = load_model(model_file)
inception_chest = load_model('inceptionv3_chest.h5')
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]="static"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Covid-19', methods=['POST'])
def upload_Covid():
    uploaded_file = request.files['file']
    full_filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], full_filename))
    path=os.path.join(app.config['UPLOAD_FOLDER'], full_filename)
    image = cv2.imread(path) # read file 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    inception_pred = inception_chest.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
      inception_chest_pred = 'COVID POSITIVE' 
    else:
       inception_chest_pred =  'COVID NEGATIVE'
    print(inception_chest_pred)
    return render_template('covid.html',filename = uploaded_file.filename, text=inception_chest_pred)



@app.route('/Pneumonia', methods=['POST'])
def upload_Pneumonia():
    uploaded_file = request.files['file']
    full_filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], full_filename))
    path=os.path.join(app.config['UPLOAD_FOLDER'], full_filename)
    img = Image.open(path) # we open the image
    img_d = img.resize((224,224))
    # we resize the image for the model
    rgbimg=None
    #We check if image is RGB or not
    if len(np.array(img_d).shape)<3:
        rgbimg = Image.new("RGB", img_d.size)
        rgbimg.paste(img_d)
    else:
        rgbimg = img_d
    rgbimg = np.array(rgbimg,dtype=np.float64)
    rgbimg = rgbimg.reshape((1,224,224,3))
    predictions = model.predict(rgbimg)
    a = int(np.argmax(predictions))
    if a==1:
       a = "pneumonic"
    else:
       a="healthy"
    return render_template('content.html',filename = uploaded_file.filename, text=a)