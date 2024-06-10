import flwr as fl
import tensorflow as tf
import json
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Initialize and compile the TensorFlow model
model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), classes=10, weights=None)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.acc_list = []  # Initialize acc_list within the class
    
    def get_parameters(self, config):
        # Return model parameters as a list of NumPy ndarrays
        return model.get_weights()

    def fit(self, parameters, config):
        # Set model parameters, train model, return updated model parameters
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        
        # Append accuracy values to acc_list
        self.acc_list.extend(history.history['accuracy'])

        with open("client_accuracy.json", "w") as f:
            json.dump(self.acc_list, f)
        
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set model parameters, evaluate model on test data, return result
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy)}


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())
