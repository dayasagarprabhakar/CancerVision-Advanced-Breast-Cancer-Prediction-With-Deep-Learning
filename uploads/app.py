from __future__ import division,print_function
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
global graph
graph=tf.Graph()
app = Flask(__name__)
model=load_model(breastcancer.h5)
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=["GET","POST"])
def uploadfile():
    if request.method == "POST":
        f = request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size(64,64))
        x=image.img_to_array(img)/255
        x=np.expand_dims(x,axis=0)
        with graph.as_default():
            preds=model.predict_classes(x)
            if preds[0][0]==0:
                text="the tumor is bening.. need  not worry!"
            else:
                text="it is a malignant... please consult doctor"
                text=text
                return text
            if __name__=='__main__':
                app.run(debug=True,threaded=False)
