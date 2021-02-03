import os
import yaml
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.random import set_seed

from dvc.api import make_checkpoint

num_classes = 10
input_shape = (28, 28, 1)
epochs = 10


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def get_model():
    set_seed(0)
    model= keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def evaluate(model, x, y):
    metrics_dict = {}
    metrics = model.evaluate(x, y, verbose=0)
    metrics_dict["loss"] = metrics[0]
    metrics_dict["acc"] = metrics[1]
    with open("metrics.yaml", "w") as f:
        yaml.dump(metrics_dict, f)


if __name__ == "__main__":
    # Load model.
    if os.path.exists("model.tf"):
        model = keras.models.load_model("model.tf")
    else:
        model = get_model()
    # Load data.
    (x_train, y_train), (x_test, y_test) = get_data()
    # Iterate over training epochs.
    for i in range(epochs):
        model.fit(x_train, y_train, batch_size=128, validation_split=0.1)
        evaluate(model, x_test, y_test)
        model.save("model.tf")
        make_checkpoint()
