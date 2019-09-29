"""
Module of parsing utilities.
"""

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import time
import csv

"""
*** Parsing functions for the word embeddings ***
"""

"""
Parse the word embeddings file. Returns a dictionary that maps words to their respective word embedding vector.
"""
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


"""
Given a index of embeddings (dictionary of word->(vector embedding)), return a word index (dictionary of word->number).
"""
def word_index(embeddings_index, remove_unknown_words=False):
    words = embeddings_index.keys()
    if remove_unknown_words:
        indexes = [i for i in range(len(embeddings_index))]
    else:
        indexes = [(i+1) for i in range(len(embeddings_index))]     # save index 0 for unknown words
    return dict(zip(words, indexes))


"""
Receives an embeddings index (the output of corpus2embeddings_index function), a word_index (dictionary of word->number)
and a vector_size, and returns an embedding matrix which may be used for creating an embedding layer
in a Keras neural network.
"""
def embedding_matrix(embeddings_index, word_index, vector_size=300, remove_unknown_words=False):
    vocab_size = len(word_index)
    assert vocab_size == len(embeddings_index)
    assert len(embeddings_index) == len(word_index)
    if remove_unknown_words:
        embedding_matrix = np.zeros((vocab_size, vector_size))
    else:
        embedding_matrix = np.zeros((vocab_size+1, vector_size))    # save index 0 for unknown words
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) == vector_size:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


"""
*** Parsing functions for the corpus files ***
"""

"""
Given a csv corpus file name, returns it as a pandas data frame
"""
def csv2dataframe(data_filename):
    return pd.read_csv(data_filename)


"""
Replace every non alphanumeric character c with the string ' c ' (the character c followed and preceded by a space.
"""
def space_non_alphanumeric(text):
    r = re.compile('([^a-zA-Z0-9ñÍÓÚÉÁ \t\n\r\f\váéíóú])')
    return r.sub(r' \1 ', text)

def conditional_lowercase(word):
    if word.isupper():
        return word.lower()
    else:
        return word

def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape(""""#$%&'()*+,/:;<=>@[\]^_`{|}~"""), '', tweet)  # remove punctuations (removed . ? - ! to check if it improves)
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    return tweet

def pre_process_tweets(tweets):
    clean_list = tweets.lower().strip().split()
    clean_list = list(map(clean_tweet, clean_list))
    return ' '.join(clean_list)

"""
Receives a corpus file name and parses it, returning 
"""
def parse_corpus(corpus_filename, word_index, max_features=35569, remove_unknown_words=False, clean_up=False):
    print('Parsing {file_name}...'.format(file_name=corpus_filename))
    df = pd.read_csv(corpus_filename)
    tokenizer = Tokenizer(num_words=max_features, filters='', lower=False, split=' ', char_level=False)
    texts_list = df['text'].tolist()
    if clean_up:
        texts_list = list(map(pre_process_tweets, texts_list))
    texts_list = list(map(space_non_alphanumeric, texts_list))
    tokenizer.fit_on_texts(texts_list)
    # list_tokenized_texts = tokenizer.texts_to_sequences(texts_list)
   
    # to check how data is being processed
    with open('corpusss.csv', 'w', newline='', encoding="utf-8") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(texts_list)

    list_tokenized_texts = []
    for i in range(len(texts_list)):
        word_list = text_to_word_sequence(texts_list[i], filters='', lower=False, split=' ')
        if remove_unknown_words:
            list_tokenized_texts.append(list(map(lambda x: word_index[x] if x in word_index else -1, word_list)))
            list_tokenized_texts[-1] = list(filter(lambda x: x != -1, list_tokenized_texts[-1]))
        else:
            list_tokenized_texts.append(list(map(lambda x: word_index[x] if x in word_index else 0, word_list)))    # save index 0 for unknown words

    max_len = 40    # on data_test.csv, the maximum number of words in a tweet is 43. So max_len=40 seems reasonable
    x = pad_sequences(list_tokenized_texts, maxlen=max_len)
    y = df['humor'].tolist()
    print('{texts_num} texts processed'.format(texts_num=len(x)))
    print('{vocab_num} different words'.format(vocab_num=len(tokenizer.word_index)))
    return x, y, tokenizer.word_index


"""
For debugging
"""
def test():
    DATA_TEST = '../corpus/data_test.csv'
    DATA_TRAIN = '../corpus/data_train.csv'
    DATA_VAL = '../corpus/data_val.csv'
    WORD_EMBEDDINGS = '../word_embedding/intropln2019_embeddings_es_300.txt'
    

    print('Test space_non_alphanumeric')
    s1 = 'Esos que dicen que lo más difícil en la vida es olvidar a alguien seguro nunca han intentado hacer un sándwich sin comer rebanadas de jamón.'
    print(s1)
    print(space_non_alphanumeric(s1))
    s2 = "- Amor, ¿me queda bien el disfraz?\n- Sí, amor, te ves bonita de vaca.\n- ¿Vaca? ¡Pero si voy de dálmata!\n#Chistes ..."
    print(s2)
    print(space_non_alphanumeric(s2))
    s3 = "Ya me empezó a dar hambre de la mala, de esa que te hace poner tuits pagados y convertir hashtags en tendencia."
    print(space_non_alphanumeric(s3))
    s4 = 'Eso no me lo esperaba 😂😂😂'
    print(space_non_alphanumeric(s4))
    s5 = 'Hay mucho portugués disfrazado...'
    print(space_non_alphanumeric(s5))
    s6 = '"- ¿Cuántas anclas tiene un barco?\n- 11\n- ¿Por qué?\n- Porque siempre dicen ""eleven anclas""\n#Chiste"'
    print(space_non_alphanumeric(s6))

    print('Test corpus2embeddins_index')
    start_time = time.time()
    embeddings_idx = embeddings_index(WORD_EMBEDDINGS)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Test word_index')
    start_time = time.time()
    words_idx = word_index(embeddings_idx)
    assert len(embeddings_idx) == len(words_idx)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Test parse_corpus on data_test.csv')
    parse_corpus(DATA_TEST, words_idx)
    print('Test parse_corpus on data_val.csv')
    parse_corpus(DATA_VAL, words_idx)
    print('Test parse_corpus on data_train.csv')
    x_train, y_train, word_index_train = parse_corpus(DATA_TRAIN, words_idx)

    print('Test embedding_matrix')
    word_idx = word_index(embeddings_idx)
    start_time = time.time()
    embedding_mat = embedding_matrix(embeddings_idx, word_idx)
    assert len(word_idx)+1 == embedding_mat.shape[0]    # save index 0 for unknown words
    assert embedding_mat.shape[1] == 300
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    test()
