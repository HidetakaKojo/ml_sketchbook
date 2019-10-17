import tensorflow as tf
from tensorflow.keras.layers import Input, Add
import sys

keras = tf.keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_labels = test_labels[:1000]
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    input_ = Input((784,))
    model1 = keras.models.load_model("./train_1.pb")
    model1._name = 'train1'
    model2 = keras.models.load_model("./train_2.pb")
    model2._name = 'train2'
    output_1 = model1(input_)
    output_2 = model2(input_)
    output_ = Add()([output_1, output_2])
    model = keras.Model(inputs=[input_], outputs=[output_])
    return model

i = int(sys.argv[1])
model = create_model()
for i in range(i, i+10):
    pred = model.predict([[test_images[i]]])
    print(pred)
    print(test_labels[i])
