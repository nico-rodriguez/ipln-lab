#!/usr/bin/env python3

from constants import *
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


def _remove_tildes(text):
    text.replace('á', 'a')
    text.replace('é', 'e')
    text.replace('í', 'i')
    text.replace('ó', 'o')
    text.replace('ú', 'u')
    text.replace('Á', 'A')
    text.replace('É', 'E')
    text.replace('Í', 'I')
    text.replace('Ó', 'O')
    text.replace('Ú', 'U')
    return text


def _get_word_embeddings(embeddings_filename):
    word_embedding = {}
    # save indexes 0 and 1 for padding and unknown words
    with open(embeddings_filename, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            word = _remove_tildes(word)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # little hack to catch 4 four words (special characters)
            if len(coefs) != 300:
                word = line[0]
                coefs = line[1:]
                coefs = np.fromstring(coefs, 'f', sep=' ')
            word_embedding[word] = coefs
    return word_embedding


def _get_embedding_vector(word_embedding, word):
    if word in word_embedding:
        return word_embedding[word]
    elif word.capitalize() in word_embedding:
        return word_embedding[word.capitalize()]
    elif word.lower() in word_embedding:
        return word_embedding[word.lower()]
    elif word.upper() in word_embedding:
        return word_embedding[word.upper()]
    else:
        return None


def _adjust_word_to_match_embedding(word_embedding, word):
    if word.capitalize() in word_embedding:
        return word.capitalize()
    elif word.lower() in word_embedding:
        return word.lower()
    elif word.upper() in word_embedding:
        return word.upper()
    else:
        return word


def embeddings_file2embeddings_matrix(embeddings_filename):
    print('Computando matriz de embeddings...')
    word_embedding = _get_word_embeddings(embeddings_filename)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=EMBEDDINGS_NUM_WORDS, filters='\n\t.:()-',
                                                      lower=False, split=' ', char_level=False, oov_token='<UNK>')
    tokenizer.fit_on_texts(list(word_embedding.keys()))
    embeddings_word_index = tokenizer.word_index
    # save index 0 and 1 for padding and unknown words
    embedding_matrix = np.zeros((len(word_embedding) + 2, EMBEDDING_VECTOR_SIZE))
    for word, index in embeddings_word_index.items():
        if index != 1:
            embedding_vector = _get_embedding_vector(word_embedding, word)
            embedding_matrix[index] = embedding_vector
    return embedding_matrix, tokenizer


def preprocess_data(texts_list, tokenizer):
    print('Pre procesando textos...')
    texts_list = list(map(lambda x: _space_non_alphanumeric(x), texts_list))
    texts_list = list(map(lambda x: _remove_tildes(x), texts_list))
    texts_list = list(map(lambda x: ' '.join(list(map(lambda w: _adjust_word_to_match_embedding(tokenizer.word_index, w), x.split(' ')))), texts_list))
    indexes_list = tokenizer.texts_to_sequences(texts_list)
    indexes_list = tf.keras.preprocessing.sequence.pad_sequences(indexes_list, maxlen=MAX_WORDS_PER_TWEET)
    return np.array(indexes_list)
