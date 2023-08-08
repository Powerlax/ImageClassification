import tensorflow as tf
import numpy as np
import pathlib


#get the dataset
data = pathlib.Path('C:/Users/innav_z3e3dq9/.keras/datasets/PetImages')
print(data)
tds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='training', seed=123, image_size=(224,224))
vds = tf.keras.utils.image_dataset_from_directory(directory=data, validation_split=0.2, subset='validation',seed=123, image_size=(224,224))
names = tds.class_names
print(names)
num = len(names)

def main(tds, vds):
    #tune it using the autotune
    tuner = tf.data.AUTOTUNE
    tds = tds.cache().shuffle(1000).prefetch(buffer_size=tuner)
    vds = vds.cache().prefetch(buffer_size=tuner)

    #add data augmentation for randomness
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224,3)),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomCrop(224, 224, 3),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        ]
    )


    #build model
    InceptionV3 = tf.keras.applications.InceptionV3(include_top= False, input_shape= (224, 224, 3), weights= 'imagenet')
    for layer in InceptionV3.layers:
       layer.trainable = False
    

    model = tf.keras.Sequential([
        data_aug,
        InceptionV3,
      tf.keras.layers.GlobalAveragePooling2D(), #global average pooling 
      tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(num, activation='softmax'),
    ])

    #compile using adam optimizer and crossentropy loss
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    #summary of model
    model.summary()

    #training
    model.fit(tds, validation_data=vds, epochs=30)     #uncomment for training

    #save model
    model.save('baseline_model.h5')

main(tds, vds)
