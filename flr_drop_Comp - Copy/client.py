import flwr as fl
import tensorflow as tf
import json
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf

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


# Assume class 0 is the minority class
import numpy as np
minority_class_label = 0
minority_indices_train = np.where(y_train == minority_class_label)[0]
num_indices_to_keep=len(minority_indices_train) // 2
indices_to_keep_train = minority_indices_train[:num_indices_to_keep]
# Remove indices corresponding to minority class label from y_train
y_train_filtered = np.delete(y_train, indices_to_keep_train, axis=0)
x_train_filtered = np.delete(x_train, indices_to_keep_train, axis=0)

minority_indices_train = np.where(y_test == minority_class_label)[0]
num_indices_to_keep=len(minority_indices_train) // 2
indices_to_keep_train = minority_indices_train[:num_indices_to_keep]
# Remove indices corresponding to minority class label from y_train
y_test_filtered = np.delete(y_test, indices_to_keep_train, axis=0)
x_test_filtered = np.delete(x_test, indices_to_keep_train, axis=0)

(x_train, y_train), (x_test, y_test) = (x_train_filtered ,y_train_filtered), (x_test_filtered ,y_test_filtered) 

class CifarClient(fl.client.NumPyClient):
    def __init__(self, initial_dropout_rate=0.5):
        super().__init__()
        self.acc_list = []
        self.dropout_rate = initial_dropout_rate
        self.last_round_last_accuracy = 0
        self.model = create_model(self.dropout_rate)  # Initialize the model here

    def get_parameters(self, config):
        # Return the model's weights
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set the model's weights
        self.model.set_weights(parameters)

        # Perform model quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()

        # Save the quantized model to a file
        with open("quantized_model.tflite", "wb") as f:
            f.write(quantized_tflite_model)

        # Update pruning step (if necessary)
        callbacks = [sparsity.UpdatePruningStep()]
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3, callbacks=callbacks)

        # Update dropout rate and record accuracy
        if len(self.acc_list) > 0:
            current_first_accuracy = self.acc_list[-1]
            if current_first_accuracy > self.last_round_last_accuracy:
                self.dropout_rate = max(0.3, self.dropout_rate - 0.05)
            else:
                self.dropout_rate = min(0.7, self.dropout_rate + 0.05)

        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3)
        self.acc_list.extend(history.history['accuracy'])

        # Save accuracy to file
        with open("client_accuracy.json", "w") as f:
            json.dump(self.acc_list, f)
        
        self.last_round_last_accuracy = history.history['accuracy'][-1]

        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set the model's weights
        self.model.set_weights(parameters)

        # Evaluate the model
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy)}

# Initialize and start the Flower client
initial_dropout_rate = 0.5
fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient(initial_dropout_rate).to_client())
