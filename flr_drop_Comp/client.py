import flwr as fl
import tensorflow as tf
import json
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
from tensorflow.keras import layers


def create_model(dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
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
        self.dropouts = []
        self.dropout_rate = initial_dropout_rate
        self.last_round_last_accuracy = 0
        self.random_var = 0
        self.model = create_model(self.dropout_rate)  # Initialize the model here

    def get_parameters(self, config):
        # Return the model's weights
        return self.model.get_weights()

    def representative_dataset_gen(self):
        for sample in x_train[:1000]:
            # Preprocess the input data to match the expected format
            sample = sample.astype('float32') / 255.0  # Normalize the pixel values to [0, 1]
            sample = tf.expand_dims(sample, axis=0)  # Add batch dimension
            yield [sample]

    def fit(self, parameters, config):
        # Set the model's weights
        self.model.set_weights(parameters)
        dropout_rate = config.get("dropout_rate", 0.5)  # Use default if not specified

        for layer in self.model.layers:
            if isinstance(layer, layers.Dropout):
                layer.rate = dropout_rate

        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, steps_per_epoch=3, validation_split=0.2)
        self.acc_list.extend(history.history['val_accuracy'])
        # Update dropout rate and record accuracy
        if len(self.acc_list) > 0:
            current_first_accuracy = self.acc_list[-1]
            if current_first_accuracy > self.last_round_last_accuracy:
                self.random_var = 1
            else:
                self.random_var = -1
        
        self.dropouts.append(self.dropout_rate)

        # Save accuracy to file
        with open("client_accuracy.json", "w") as f:
            json.dump(self.acc_list, f)

        with open("dropout_rate.json", "w") as fa:
            json.dump(self.dropouts, fa)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]  # Use int8 quantization
        converter.representative_dataset = self.representative_dataset_gen
        quantized_tflite_model = converter.convert()

        
        self.last_round_last_accuracy = history.history['val_accuracy'][-1]

        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set the model's weights
        self.model.set_weights(parameters)

        # Evaluate the model
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy), "random_var": self.random_var}

# Initialize and start the Flower client
initial_dropout_rate = 0.5
fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient(initial_dropout_rate).to_client())