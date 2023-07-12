import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow_datasets as tfds
import pathlib

#get the dataset
data = tf.keras.utils.get_file('Images', origin='http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', untar=True)
data = pathlib.Path(data)
tds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='training', seed=123, image_size=(180,180))
vds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='validation',seed=123, image_size=(180,180))
names = tds.class_names
num = len(names)

#tune it using the autotune
tuner = tf.data.AUTOTUNE
tds = tds.cache().shuffle(1000).prefetch(buffer_size=tuner)
vds = vds.cache().prefetch(buffer_size=tuner)

#build model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=1./255, input_shape=(180,180,3)),   #to rescale the 0, 255 value range down to 0, 1
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),   #first cnn with relu activiation function
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),                      #Standard Dense NN
    tf.keras.layers.Dense(num)                                          #one neuron for each class
])

#compile using adam optimizer and crossentropy loss
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#summary of model
model.summary()

#training
training_history=model.fit(tds, validation_data=vds, epochs=10)

















'''
print(data)

train = tfds.load('stanford_dogs', split='test')
test = tfds.load('stanford_dogs', split='test')

for ex in train.take(4):
    print(ex)
'''





#My attempt at load a tfds dataset
#builder = tfds.builder(name='cats_vs_dogs')
#ds = builder.as_dataset(split='test', as_supervised=True)
#tds, vds = ds["train"], ds["test"]
#print(ds)
