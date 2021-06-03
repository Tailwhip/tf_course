#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# Load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Show data
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# Data normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# Model definition
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Model compilation
tf.optimizers.SG
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Model evaluation
model.evaluate(test_images, test_labels)


# Assesment
# # GRADED FUNCTION: train_mnist
# def train_mnist():
#     # Please write your code only where you are indicated.
#     # please do not remove # model fitting inline comments.

#     # YOUR CODE SHOULD START HERE
#     class myCallback(tf.keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs={}):
#             if(logs.get('acc')>0.99):
#               print("\nReached 99% accuracy so cancelling training!")
#               self.model.stop_training = True

#     callbacks = myCallback()
#     # YOUR CODE SHOULD END HERE

#     mnist = tf.keras.datasets.mnist

#     (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
#     # YOUR CODE SHOULD START HERE
#     x_train = x_train / 255.0
#     x_test = x_test / 255.0
#     # YOUR CODE SHOULD END HERE
#     model = tf.keras.models.Sequential([
#         # YOUR CODE SHOULD START HERE
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation=tf.nn.relu),
#         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#         # YOUR CODE SHOULD END HERE
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     # model fitting
#     history = model.fit(# YOUR CODE SHOULD START HERE
#         x_train, y_train, epochs=10, callbacks=[callbacks]
#         # YOUR CODE SHOULD END HERE
#     )
#     # model fitting
#     return history.epoch, history.history['acc'][-1]