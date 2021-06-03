#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=5000)

print(model.predict([10.0]))


# Assesment
# def house_model(y_new):
#     xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)# Your Code Here#
#     ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)# Your Code Here#
#     model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])# Your Code Here#
#     model.compile(optimizer='sgd', loss='mean_squared_error')# Your Code Here#)
#     model.fit(xs, ys, epochs=5000)# Your Code here#)
#     return model.predict(y_new)[0]
