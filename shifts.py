%matplotlib inline

import os
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

print('Using TensorFlow', tf.__version__)

generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = [-100,-50,0,50,100], height_shift_range = [-50,0,50])

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));