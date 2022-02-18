# Quickstart example from: https://flower.dev/docs/quickstart_tensorflow.html
import flwr as fl
import tensorflow as tf

# load in training and test data from CIFAR10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# compile a MobilNetV2  model from TF?Keras, 10 output classes
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

'''
Create client class with the flwr.client interface. The server will use this class to interact with the client. When
clients are selected for training, the server sends training instruction to the client. When this data is received, then
the client will call one this class' methods.

The NumPy client makes it easier to implement the Client interface when using Keras. 
'''
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


'''
Create an instance of CifarClient and run it.

The string "[::]:8080" tells the client which server to connect to.
'''
fl.client.start_numpy_client("[::]:8080", client=CifarClient())