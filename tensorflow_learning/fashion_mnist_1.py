# https://thecleverprogrammer.com/2021/03/17/neural-networks-in-machine-learning/

import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_valid, x_train = x_train[:5000]/255.0, x_train[5000:]/255.0
y_valid, y_train = y_train[:5000], y_train[5000:]

classes = ["T-shirt/top", "Trouser", "Pullover", 
           "Dress", "Coat", "Sandal", "Shirt", 
           "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.legend()
plt.show()

import numpy as np
x_new = x_test[:3]
y_pred = model.predict_classes(x_new)
print(y_pred)
print(np.array(classes)[y_pred])