#!/usr/bin/env python3

from metrics import Metrics
import numpy as np
from hyperparameters import *
import tensorflow as tf
import time


def compile_classifier():
    print('Compilando clasificador...')
    return tf.keras.models.Sequential()


def train_model(model, x_train, y_train, x_val, y_val):
    print('Entrenando modelo...')
    metrics = Metrics()
    start_time = time.time()
    # model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[metrics],
    #           workers=2, use_multiprocessing=True, validation_freq=1, validation_data=(x_val, y_val))
    end_time = time.time()
    print('Entrenamiento finalizado en ({min} minutos)'.format(min=(end_time - start_time) / 60))


def evaluate_model(model, x_test, y_test):
    print('Evaluando modelo...')
    # model.evaluate(x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)


def test_model(model, x_test):
    print('Testeando modelo...')
    # return np.asarray(model.predict_classes(x_test, batch_size=BATCH_SIZE))
    return np.array([1, 0, 1])
