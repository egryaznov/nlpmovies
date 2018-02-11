# IMDB Movie Review Sentiment Classification
# Second Assignment Solution
# NLP Course, Innopolis University, Spring 2017
# Author: Evgeny Gryaznov


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt


def plot_loss(history):
    """ Plots the values of a loss function through training time (epoches). """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss of the Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_accuracy(history):
    """ Plots the accuracy of a model through training time (epoches). """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy of the Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


amount_of_first_top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=amount_of_first_top_words)
# Show samples of dataset
print('Review representation sample: %s' % x_train[0])
print('Class representation sample: %s' % y_train[0])
# Shrink and truncate the data so it can fit into LSTM
max_review_length = 600
x_test  = sequence.pad_sequences(x_test, maxlen=max_review_length)
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
# Compile the model
model = Sequential()
embedding_vector_length = 32
model.add(Embedding(amount_of_first_top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Train the model and plot performance
history = model.fit(x_train, y_train, validation_split=0.33, epochs=3, batch_size=64)
plot_loss(history)
plot_accuracy(history)
# Evaluate the accuary of the model on the test data
assess = model.evaluate(x_test, y_test, verbose=0)
print("Evaluated accuracy: %.2f%%" % (assess[1] * 100))
