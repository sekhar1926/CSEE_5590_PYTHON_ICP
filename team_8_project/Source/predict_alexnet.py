from __future__ import division
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as im

model = load_model('disease_alexnet_detector.h5')

path = '/Users/adisekharrajuuppalapati/PycharmProjects/plant_diseases/static/9bd4db81-94ef-4a9e-a44a-e335fb372691___Matt.S_CG 7786.JPG'


img = plt.imread(path)
img = np.array(img)
img = img / 255.0
plt.imshow(img)
plt.show()


img = image.load_img(path, target_size=(224,224))
img = image.img_to_array(img)
img = img / 255.0
img = np.array(img)
plt.imshow(img)


img = img.reshape((1,224,224,3))

ans = model.predict(img).argmax()
print(ans)

plt.title(ans)
plt.show()