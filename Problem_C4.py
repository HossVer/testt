# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import keras.callbacks
import tensorflow as tf
import numpy as np
import urllib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class  myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')> 0.76 and logs.get('val_accuracy')>0.76:
            print("\nAlready reached teh desired accuracy, stop training!!!")
            self.model.stop_training=True

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    with open("./sarcasm.json", 'r') as f:
      datastore = json.load(f)

    for item in datastore:
      sentences.append(item['headline'])
      labels.append(item['is_sarcastic'])

    training_sentences = sentences[0:training_size]
    training_labels = labels[0:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_idex = tokenizer.word_index

    #--- The text into sequence and pad it
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences,
                                    truncating=trunc_type,
                                    maxlen=max_length,
                                    padding=padding_type)

    validaton_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validaton_sequences,
                                      truncating=trunc_type,
                                      maxlen= max_length,
                                      padding=padding_type)

    #--- define the training label
    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size ,embedding_dim, input_length =max_length),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    callbacks = myCallback()
    model.compile(loss= 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    model.fit(training_padded,
              training_labels,
              validation_data=(
                  validation_padded,
                  validation_labels),
              epochs = 10,
              verbose = 2,
              callbacks = callbacks
              )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
