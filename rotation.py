%matplotlib inline

import os
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

print('Using TensorFlow', tf.__version__)

generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40)

image_path = 'images/train/cat/cat.jpg'

plt.imshow(plt.imread(image_path));

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));