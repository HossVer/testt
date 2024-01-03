# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile

import keras.callbacks
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

class myCallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')> 85:
            print("Stopping training")
            self.model.stop_training=True


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"

    training_datagen = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        validation_split=0.2)


    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator= training_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size = 32,
        class_mode = 'categorical',
        target_size = (150,150),
        subset ='training'
    )

    validation_datagen = ImageDataGenerator(
        rescale =1/255
    )

    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size= 32,
        class_mode='categorical',
        target_size=(150,150),
        subset = 'validation'
    )

    model=tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


    callbacks = myCallbacks()
    model.compile(optimizer = 'adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_generator,
              validation_data = validation_generator,
              epochs = 20,
              steps_per_epoch = 20,
              verbose = 1,
              callbacks = callbacks,
              validation_steps = 3)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
