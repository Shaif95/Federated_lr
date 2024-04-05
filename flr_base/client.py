import flwr as fl
import tensorflow as tf
import json
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def create_model(dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.acc_list = []  # Initialize acc_list within the class
        self.dropout_rate = .5
        self.model = create_model(self.dropout_rate)  # Initialize the self.model here
    
    def get_parameters(self, config):
        # Return self.model parameters as a list of NumPy ndarrays
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set self.model parameters, train self.model, return updated self.model parameters
        self.model.set_weights(parameters)
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        
        # Append accuracy values to acc_list
        self.acc_list.extend(history.history['accuracy'])

        with open("client_accuracy.json", "w") as f:
            json.dump(self.acc_list, f)
        
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set self.model parameters, evaluate self.model on test data, return result
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy)}


# Initialize and start the Flower client
initial_dropout_rate = 0.5
fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())
