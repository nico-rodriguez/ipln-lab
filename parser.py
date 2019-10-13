#!/usr/bin/env python3

from constants import *
from hyperparameters import *
from nltk import download
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


download('stopwords')


def embeddings_index(word_embedding_filename):
    print('Indexing word vectors...')
    embeddings_index = {}
    with open(word_embedding_filename, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def word_index(embeddings_index):
    words = embeddings_index.keys()
    indexes = [i for i in range(len(embeddings_index))]
    return dict(zip(words, indexes))


def embedding_matrix(embeddings_index, word_index):
    vocab_size = len(word_index)
    assert vocab_size == len(embeddings_index)
    assert len(embeddings_index) == len(word_index)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_VECTOR_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) == EMBEDDING_VECTOR_SIZE:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def csv2dataframe(data_filename):
    return pd.read_csv(data_filename)


def space_non_alphanumeric(text):
    r = re.compile('([^a-zA-Z0-9ñÍÓÚÉÁ \t\n\r\f\váéíóú])')
    return r.sub(r' \1 ', text)


def remove_stop_words(text):
    stop_words = stopwords.words('spanish')
    if text in stop_words:
        return None
    else:
        return text


def parse_corpus(corpus_filename, word_index):
    print('Parsing {file_name}...'.format(file_name=corpus_filename))
    df = pd.read_csv(corpus_filename)
    tokenizer = Tokenizer(num_words=EMBEDDINGS_NUM_WORDS, filters='', lower=False, split=' ', char_level=False)
    texts_list = df['text'].tolist()
    texts_list = list(map(space_non_alphanumeric, texts_list))
    tokenizer.fit_on_texts(texts_list)

    list_tokenized_texts = []
    for i in range(len(texts_list)):
        word_list = text_to_word_sequence(texts_list[i], filters='', lower=False, split=' ')
        word_list = list(map(remove_stop_words, word_list))
        word_list = [elem for elem in word_list if elem]
        list_tokenized_texts.append(list(map(lambda x: word_index[x] if x in word_index else -1, word_list)))
        list_tokenized_texts[-1] = list(filter(lambda x: x != -1, list_tokenized_texts[-1]))

    x = pad_sequences(list_tokenized_texts, maxlen=MAX_WORDS_PER_TWEET)
    y = np.array(df['humor'].tolist())
    print('{texts_num} texts processed'.format(texts_num=len(x)))
    return x, y, tokenizer.word_index
