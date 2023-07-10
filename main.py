import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow_datasets as tfds
import pathlib

data = tf.keras.utils.get_file('flower_photos', origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data = pathlib.Path(data)
img_count = len(list(data.glob('*/*.jpg')))
print(img_count)







#My attempt at load a tfds dataset
#builder = tfds.builder(name='cats_vs_dogs')
#ds = builder.as_dataset(split='test', as_supervised=True)
#tds, vds = ds["train"], ds["test"]
#print(ds)
