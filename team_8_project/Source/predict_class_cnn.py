

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('Disease_detector_model.h5')

path = '/Users/adisekharrajuuppalapati/PycharmProjects/plant_diseases/static/4e770559-e332-470e-aae3-c7ad0399dd20___FREC_Scab 3513.JPG'

img = plt.imread(path)
img = np.array(img)
img = img/255.0
plt.imshow(img)
plt.show()
img = image.load_img(path, target_size=(28,28))
img = image.img_to_array(img)
img = img/255.0
img = np.array(img)
plt.imshow(img)

img = img.reshape((1,28,28,3))

print(img.shape)

ans = model.predict(img).argmax()

print(ans)
plt.title(ans)
plt.show()