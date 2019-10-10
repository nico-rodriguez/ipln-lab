#!/usr/bin/env python3

from hyperparameters import *
import numpy as np
import pandas as pd
import re
import tensorflow as tf


def load_data(data_filename):
    print('Leyendo archivo {filename}...'.format(filename=data_filename))
    df = pd.read_csv(data_filename)
    return df['text'].to_numpy(), df['humor'].to_numpy()


def _space_non_alphanumeric(text):
    r = re.compile('([^a-zA-Z0-9ñÍÓÚÉÁ \t\n\r\f\váéíóú])')
    return r.sub(r' \1 ', text)


def preprocess_data(texts_list):
    print('Pre procesando textos...')
    texts_list = list(map(lambda x: _space_non_alphanumeric(x), texts_list))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=35569, filters='\n\t.:()-', lower=False, split=' ',
                                                      char_level=False, oov_token='<UNK>')
    tokenizer.fit_on_texts(texts_list)
    indexes_list = tokenizer.texts_to_sequences(texts_list)
    indexes_list = tf.keras.preprocessing.sequence.pad_sequences(indexes_list, maxlen=MAX_WORDS_PER_TWEET)
    return np.array(indexes_list), tokenizer.word_index


def embeddings_file2embeddings_matrix(embeddings_filename, word_index):
    print('Computando matriz de embeddings...')
    VOCABULARY_SIZE = len(word_index)
    EMBEDDING_VECTOR_SIZE = 300
    # save indexes 0 and 1 for padding and unknown words
    embedding_matrix = np.zeros((VOCABULARY_SIZE+2, EMBEDDING_VECTOR_SIZE))
    with open(embeddings_filename, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # little hack to catch 4 four words (special characters)
            if len(coefs) != 300:
                word = line[0]
                coefs = line[1:]
                coefs = np.fromstring(coefs, 'f', sep=' ')
            if word_index.get(word) is not None:
                index = word_index[word]
                embedding_matrix[index] = coefs
            elif word_index.get(word.capitalize()) is not None:
                index = word_index[word.capitalize()]
                embedding_matrix[index] = coefs
            elif word_index.get(word.lower()) is not None:
                index = word_index[word.lower()]
                embedding_matrix[index] = coefs
            elif word_index.get(word.upper()) is not None:
                index = word_index[word.upper()]
                embedding_matrix[index] = coefs
    return embedding_matrix
