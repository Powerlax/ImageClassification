import tensorflow as tf
import numpy as np
import pathlib

#get the dataset
data = tf.keras.utils.get_file('Images', origin='http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', untar=True)
#data = tf.keras.utils.get_file('flower_photos', origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar=True)
data = pathlib.Path(data)
tds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='training', seed=123, image_size=(180,180))
vds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='validation',seed=123, image_size=(180,180))
names = tds.class_names
num = len(names)

#tune it using the autotune
tuner = tf.data.AUTOTUNE
tds = tds.cache().shuffle(1000).prefetch(buffer_size=tuner)
vds = vds.cache().prefetch(buffer_size=tuner)

#add data augmentation for randomness
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(180, 180,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)

#build model
model = tf.keras.Sequential([
    data_aug,
    tf.keras.layers.Rescaling(scale=1./255, input_shape=(180,180,3)),   #to rescale the 0, 255 value range down to 0, 1
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),   #first cnn with relu activiation function
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),                      #Standard Dense NN
    tf.keras.layers.Dense(num)                                          #one neuron for each class
])

#compile using adam optimizer and crossentropy loss
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#summary of model
model.summary()

#training
history=model.fit(tds, validation_data=vds, epochs=15)

#predict
sunflower_url = "https://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(180, 180)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(names[np.argmax(score)], 100 * np.max(score))
)

#save model
model.save('model/dogs')

