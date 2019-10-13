#!/usr/bin/env python3

from metrics import Metrics
import numpy as np
from hyperparameters import *
from keras.layers import Dense, Embedding, GRU
from keras.models import Sequential
import time


def compile_classifier(embedding_matrix):
    print('Compilando clasificador...')
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                        input_length=MAX_WORDS_PER_TWEET, weights=[embedding_matrix], trainable=False, mask_zero=True))
    model.add(GRU(units=TOTAL_UNITS, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                  kernel_initializer=KERNEL_INITIALIZER, activation=ACTIVATION))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    print('Entrenando modelo...')
    metrics = Metrics()
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, workers=2, use_multiprocessing=True,
              validation_freq=1, validation_data=(x_val, y_val), callbacks=[metrics], verbose=1)
    end_time = time.time()
    print('Entrenamiento finalizado en ({min} minutos)'.format(min=(end_time - start_time) / 60))


def evaluate_model(model, x_test, y_test):
    print('Evaluando modelo...')
    y_out = np.asarray(model.predict_classes(x_test, batch_size=BATCH_SIZE, verbose=1))
    # TODO: compare y_out with y_test


def test_model(model, x_test):
    print('Testeando modelo...')
    y_out = np.asarray(model.predict_classes(x_test, batch_size=BATCH_SIZE, verbose=1))
    return y_out
