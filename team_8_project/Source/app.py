import os

from keras.models import load_model
from keras.preprocessing import image 
from flask import Flask, render_template ,request
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt




app = Flask(__name__)

#this is give working directory of the folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
	return render_template("upload.html")

@app.route("/predictions", methods=["POST"])
def predict():
	K.clear_session()
	model = load_model('disease_detector_model.h5')
	target = os.path.join(APP_ROOT,'static')
	print(target)
	imgtemp = []
	ans = []
	
	if not os.path.isdir(target):
		os.mkdir(target)

	
	# for file in request.files['file']:


	for file in request.files.getlist("file"):
		print(file)
		filename = file.filename
		imgtemp.append(filename)
		destination = "/".join([target,filename])
		print(destination)
		file.save(destination)
		
		img = image.load_img(destination, target_size=(28,28))
		img = image.img_to_array(img)
		img = img/255.0
		img = np.array(img)
		plt.imshow(img)
		img = img.reshape((1,28,28,3))
		temp = model.predict(img).argmax()
		ans.append([temp,filename])

	return render_template("result.html",ans = ans, image_names = imgtemp )

if __name__ == '__main__':
	app.run( debug = True , host='0.0.0.0')