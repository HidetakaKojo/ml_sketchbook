import tensorflow as tf

keras = tf.keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])
        
    model.compile(optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
        )
            
    return model

model = create_model()
model.summary()


model.fit(train_images, train_labels, epochs=100,
    validation_data = (test_images, test_labels),
    )
model.save("./train_2.pb")
