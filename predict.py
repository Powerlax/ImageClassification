import tensorflow as tf
from generator import names

model = new_model = tf.keras.models.load_model('model/dogs', compile=True)
model.summary()

def predict(url):
    sunflower_path = tf.keras.utils.get_file('dog', origin=url)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    print('yes')
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(names[np.argmax(score)], 100 * np.max(score))
    )
    