# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.callbacks

#---Callback Class to stop at validation accuracy > 83%
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')> 0.84):
            print("Already reached the desired validation accuracy, stopping training")
            self.model.stop_training = True


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # DO NOT CHANGE THIS CODE
    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    val_labels_final = np.array(testing_labels)

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    index_of_words = tokenizer.word_index

    #---Make the sequence of tokenized word
    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    val_sequences = tokenizer.texts_to_sequences(testing_sentences)

    #---Pad the tokenized sequence
    padded_training = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)
    padded_validation = pad_sequences(val_sequences, maxlen=max_length)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, 'relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #---Compile the model
    callback = myCallback()
    model.compile(
        loss= 'binary_crossentropy',
        optimizer= 'adam',
        metrics=['accuracy'])

    model.fit(
        padded_training,
        training_labels_final,
        batch_size = 128,
        epoch =10,
        validation_data=(
            padded_validation,
            val_labels_final
        ),
        callbacks=callback
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save("model_A4.h5")
