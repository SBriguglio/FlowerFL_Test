# Quickstart example from: https://flower.dev/docs/quickstart_scikitlearn.html
import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# utils.py contains different functions defining ML basics
import utils


# Load MNIST dataset from OpenML
if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

# Define Logistic Regression model, initialize with utils.set_initial_params
model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
)

utils.set_initial_params(model)

""" 
Flower server interacts through the client interface. Again we use the convenience class called NumPyClient which makes
 it easier to implement the client interface when using scikit-learn. 
 
 Implementing NumPyClient usually means defining the following methods:
1. get_parameters: return the model weight as a list of NumPy ndarrays
2. set_parameters: (optional) update the local model weights with the parameters received from the server
3. fit: set the local model weights, train the local model, receive the updated local model weights
4. evaluate: test the local model

Implemented as follows...
"""
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['rnd']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# Start an instance of our MnistClient (will need to change the server address we pass to client if using another device
fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())