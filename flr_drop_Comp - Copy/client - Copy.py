import flwr as fl
import tensorflow as tf
import json
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantize

def create_model(dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class CifarClient(fl.client.NumPyClient):
    def __init__(self, initial_dropout_rate=0.5):
        super().__init__()
        self.acc_list = []
        self.dropout_rate = initial_dropout_rate
        self.last_round_last_accuracy = 0
        self.model = create_model(self.dropout_rate)  # Initialize the model here

    def get_parameters(self, config):
        # Quantize weights using TensorFlow Model Optimization Toolkit
        q_model = quantize.quantize_model(self.model)
        compressed_weights = [w.numpy() for w in q_model.trainable_weights]
        return compressed_weights

    def fit(self, parameters, config):
        decompressed_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in parameters]
        self.model.set_weights(decompressed_weights)

        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)

        self.acc_list.extend(history.history['accuracy'])
        
        with open("client_accuracy.json", "w") as f:
            json.dump(self.acc_list, f)
        
        self.last_round_last_accuracy = history.history['accuracy'][-1]

        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        decompressed_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in parameters]
        self.model.set_weights(decompressed_weights)
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy)}


# Initialize and start the Flower client
initial_dropout_rate = 0.5
fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient(initial_dropout_rate).to_client())
