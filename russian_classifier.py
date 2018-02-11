# IMDB Movie Review Sentiment Classification
# Second Assignment Solution
# NLP Course, Innopolis University, Spring 2017
# Author: Evgeny Gryaznov

import numpy
import ru_otzyv as ru
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def compile_model(topn=20000, max_review_length=300, embedding_vector_length=300, dropout_value=0.3):
    """ Builds, compiles and trains the LSTM model for russian moive review
    classification problem.
    Keyword arguments:
        params -- a dictionary of parameters for the model. Currently the
        following entries are suported:
            'top_words' -- the maximal length of a vocabulary
            'max_review_length' -- the maximal length of a review
            'embedding_vector_length' -- the length of the input vectors after
            applying `embedding` techique.
            'dropout_value' -- the percentage of units that will be dropped.
    Returns:
        A tuple: [model, history], where `model` is created model and `history`
        its history of epoches."""
# Fix random seed for reproducibility...
    numpy.random.seed(7)
# Compiling the model...
    model = Sequential()
    model.add(Embedding(topn, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100, dropout_W=dropout_value, dropout_U=dropout_value))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, nb_epochs=5):
    history = model.fit(x_train, y_train, validation_split=0.33, epochs=nb_epochs, batch_size=64)
    return history


# Final evaluation of the model
def evaluate(model, x_test, y_test):
    """ Evaluates model on the given test data and returns specified metrics.
        Keyword arguments:
            model -- trained LSTM model.
            x_test -- padded and cooked test review data.
            y_test -- padded and cooked test rating data.
        Returns:
            A tuple of scores.
    """
    scores = model.evaluate(x_test, y_test, verbose=0)
    return scores


def predict(model, review_filename, vocab):
    """ Predicts the rating of the given review.
        Keyword arguments:
            model -- trained LSTM model that will do the prediction.
            rivew_filename -- a name of the file where the text of the review
            is stored.
            vocab -- a compiled vocabulary of Russian tokens extracted from the
                dataset.
        Returns:
            The predicted rating of the review.
    """
    review = ''
    with open('sample-reviews/' + review_filename, 'r') as f:
        review = f.read()
    x = sequence.pad_sequences([ru.digitize(review, vocab)], maxlen=300)
    predicted_rating = model.predict(x)
    return predicted_rating


def build_and_evaluate(topn=20000, max_review_length=300):
    """ Run this function to compile, train, evaluate and assess our LSTM
    model in one shot!
    Returns:
        Completed LSTM that you can play with.
    """
# Load the dataset but only keep the top n words, discarding others
    print('Preparing the dataset...')
    x_train, y_train, x_test, y_test = ru.cook_data(topn=topn)
    print('    Padding sequences...')
# truncate and pad input sequences so they can fit into LSTM layer
    x_test  = sequence.pad_sequences(x_test, maxlen=max_review_length)
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
# Compile and train our LSTM
    print('Dataset preparation complete!\nCompiling the model...')
    my_lstm = compile_model(topn=topn, max_review_length=max_review_length)
    print('Mode compilation complete!\nTraining the model...')
    history = train_model(my_lstm, x_train, y_train, nb_epochs=4)
# Plot the history of training
    print('Model training complete!\nEvaluating performance...')
    plot_loss(history)
    plot_accuracy(history)
# Evaluate the accuracy of our model
    scores = evaluate(my_lstm, x_test, y_test)
    print("Final Test Data Accuracy: %.2f%%" % (scores[1] * 100))
    return my_lstm


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


my_lstm = build_and_evaluate(topn=20000)
print('-' * 30)
print('Loading vocabulary...')
vocab = ru.load('ru-vocab.json')
# Play with the model a little...
review_filename = 'positive_review0.txt'
print('Starting prediction...')
predicted_rating = predict(my_lstm, review_filename, vocab)
print('Predicted rating for this review is: ' + str(predicted_rating))


# batch normalization -- ??
# проверить распределение данных -- DONE
# балансировка данных: дублирование сэмплов -- DONE, Acc + 2%
# validation set -- ??
# голосование алгоритмов -- не буду делать
# TODO -- поменьше тренировку, побольше test: 70 на 30
# seaborn -- OK
# return subsequences true -- ??
# TODO -- softmax -- categorical crossentropy
# TODO -- RMSprop
