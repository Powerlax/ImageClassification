import tensorflow as tf
from generator import names, vds
import numpy as np

model = new_model = tf.keras.models.load_model('baseline_model.h5', compile=True)
model.summary()

def predict(url):
    path = tf.keras.utils.get_file(origin=url)

    img = tf.keras.utils.load_img(
        path, target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(names[np.argmax(score)], 100 * np.max(score))
    )

cont = True
while cont:
    url = input("Enter image url: ")
    predict(url)
    cont = input("Continue? (y/n) ") == 'y'
