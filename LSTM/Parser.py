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
import nltk
from operator import itemgetter
"""
*** Parsing functions for the word embeddings ***
"""

max_dist = 2
quota = 200 #50 #150 #200 #100
already_computed = {}

def get_nearest(target, tokenizer, word_list_tmp):
    global quota
    if quota == 0:
        return 0
    if target in already_computed:
        return (word_list_tmp[already_computed[target]])
    else:
        non_word = re.findall(r"[^a-zA-Zäáàëéèíìöóòúùñ]", target)
        if non_word:
            return 0
        else:
            quota = quota - 1
            if quota == 0:
                print('no more fixed words')
            candidates = [w if nltk.edit_distance(target,w) < max_dist else None for w in word_list_tmp.keys()]
            candidates = [c for c in candidates if c]
            candidates = [(c, tokenizer.word_counts[c]) for c in candidates if c in tokenizer.word_counts]
            if candidates:
                already_computed[target] = max(candidates,key=itemgetter(1))[0]
                return (word_list_tmp[already_computed[target]])

            else:
                return 0

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

def remove_stop_words(text):
    stop_words = {'0': '0',
    '1': '0',
    '2': '0',
    '3': '0',
    '4': '0',
    '5': '0',
    '6': '0',
    '7': '0',
    '8': '0',
    '9': '0',
    '_': '0',
    'a': '0',
    'actualmente': '0',
    'acuerdo': '0',
    'adelante': '0',
    'ademas': '0',
    'además': '0',
    'adrede': '0',
    'afirmó': '0',
    'agregó': '0',
    'ahi': '0',
    'ahora': '0',
    'ahí': '0',
    'al': '0',
    'algo': '0',
    'alguna': '0',
    'algunas': '0',
    'alguno': '0',
    'algunos': '0',
    'algún': '0',
    'alli': '0',
    'allí': '0',
    'alrededor': '0',
    'ambos': '0',
    'ampleamos': '0',
    'antano': '0',
    'antaño': '0',
    'ante': '0',
    'anterior': '0',
    'antes': '0',
    'apenas': '0',
    'aproximadamente': '0',
    'aquel': '0',
    'aquella': '0',
    'aquellas': '0',
    'aquello': '0',
    'aquellos': '0',
    'aqui': '0',
    'aquél': '0',
    'aquélla': '0',
    'aquéllas': '0',
    'aquéllos': '0',
    'aquí': '0',
    'arriba': '0',
    'arribaabajo': '0',
    'aseguró': '0',
    'asi': '0',
    'así': '0',
    'atras': '0',
    'aun': '0',
    'aunque': '0',
    'ayer': '0',
    'añadió': '0',
    'aún': '0',
    'b': '0',
    'bajo': '0',
    'bastante': '0',
    'bien': '0',
    'breve': '0',
    'buen': '0',
    'buena': '0',
    'buenas': '0',
    'bueno': '0',
    'buenos': '0',
    'c': '0',
    'cada': '0',
    'casi': '0',
    'cerca': '0',
    'cierta': '0',
    'ciertas': '0',
    'cierto': '0',
    'ciertos': '0',
    'cinco': '0',
    'claro': '0',
    'comentó': '0',
    'como': '0',
    'con': '0',
    'conmigo': '0',
    'conocer': '0',
    'conseguimos': '0',
    'conseguir': '0',
    'considera': '0',
    'consideró': '0',
    'consigo': '0',
    'consigue': '0',
    'consiguen': '0',
    'consigues': '0',
    'contigo': '0',
    'contra': '0',
    'cosas': '0',
    'creo': '0',
    'cual': '0',
    'cuales': '0',
    'cualquier': '0',
    'cuando': '0',
    'cuanta': '0',
    'cuantas': '0',
    'cuanto': '0',
    'cuantos': '0',
    'cuatro': '0',
    'cuenta': '0',
    'cuál': '0',
    'cuáles': '0',
    'cuándo': '0',
    'cuánta': '0',
    'cuántas': '0',
    'cuánto': '0',
    'cuántos': '0',
    'cómo': '0',
    'd': '0',
    'da': '0',
    'dado': '0',
    'dan': '0',
    'dar': '0',
    'de': '0',
    'debajo': '0',
    'debe': '0',
    'deben': '0',
    'debido': '0',
    'decir': '0',
    'dejó': '0',
    'del': '0',
    'delante': '0',
    'demasiado': '0',
    'demás': '0',
    'dentro': '0',
    'deprisa': '0',
    'desde': '0',
    'despacio': '0',
    'despues': '0',
    'después': '0',
    'detras': '0',
    'detrás': '0',
    'dia': '0',
    'dias': '0',
    'dice': '0',
    'dicen': '0',
    'dicho': '0',
    'dieron': '0',
    'diferente': '0',
    'diferentes': '0',
    'dijeron': '0',
    'dijo': '0',
    'dio': '0',
    'donde': '0',
    'dos': '0',
    'durante': '0',
    'día': '0',
    'días': '0',
    'dónde': '0',
    'e': '0',
    'ejemplo': '0',
    'el': '0',
    'ella': '0',
    'ellas': '0',
    'ello': '0',
    'ellos': '0',
    'embargo': '0',
    'empleais': '0',
    'emplean': '0',
    'emplear': '0',
    'empleas': '0',
    'empleo': '0',
    'en': '0',
    'encima': '0',
    'encuentra': '0',
    'enfrente': '0',
    'enseguida': '0',
    'entonces': '0',
    'entre': '0',
    'era': '0',
    'erais': '0',
    'eramos': '0',
    'eran': '0',
    'eras': '0',
    'eres': '0',
    'es': '0',
    'esa': '0',
    'esas': '0',
    'ese': '0',
    'eso': '0',
    'esos': '0',
    'esta': '0',
    'estaba': '0',
    'estabais': '0',
    'estaban': '0',
    'estabas': '0',
    'estad': '0',
    'estada': '0',
    'estadas': '0',
    'estado': '0',
    'estados': '0',
    'estais': '0',
    'estamos': '0',
    'estan': '0',
    'estando': '0',
    'estar': '0',
    'estaremos': '0',
    'estará': '0',
    'estarán': '0',
    'estarás': '0',
    'estaré': '0',
    'estaréis': '0',
    'estaría': '0',
    'estaríais': '0',
    'estaríamos': '0',
    'estarían': '0',
    'estarías': '0',
    'estas': '0',
    'este': '0',
    'estemos': '0',
    'esto': '0',
    'estos': '0',
    'estoy': '0',
    'estuve': '0',
    'estuviera': '0',
    'estuvierais': '0',
    'estuvieran': '0',
    'estuvieras': '0',
    'estuvieron': '0',
    'estuviese': '0',
    'estuvieseis': '0',
    'estuviesen': '0',
    'estuvieses': '0',
    'estuvimos': '0',
    'estuviste': '0',
    'estuvisteis': '0',
    'estuviéramos': '0',
    'estuviésemos': '0',
    'estuvo': '0',
    'está': '0',
    'estábamos': '0',
    'estáis': '0',
    'están': '0',
    'estás': '0',
    'esté': '0',
    'estéis': '0',
    'estén': '0',
    'estés': '0',
    'ex': '0',
    'excepto': '0',
    'existe': '0',
    'existen': '0',
    'explicó': '0',
    'expresó': '0',
    'f': '0',
    'fin': '0',
    'final': '0',
    'fue': '0',
    'fuera': '0',
    'fuerais': '0',
    'fueran': '0',
    'fueras': '0',
    'fueron': '0',
    'fuese': '0',
    'fueseis': '0',
    'fuesen': '0',
    'fueses': '0',
    'fui': '0',
    'fuimos': '0',
    'fuiste': '0',
    'fuisteis': '0',
    'fuéramos': '0',
    'fuésemos': '0',
    'g': '0',
    'general': '0',
    'gran': '0',
    'grandes': '0',
    'gueno': '0',
    'h': '0',
    'ha': '0',
    'haber': '0',
    'habia': '0',
    'habida': '0',
    'habidas': '0',
    'habido': '0',
    'habidos': '0',
    'habiendo': '0',
    'habla': '0',
    'hablan': '0',
    'habremos': '0',
    'habrá': '0',
    'habrán': '0',
    'habrás': '0',
    'habré': '0',
    'habréis': '0',
    'habría': '0',
    'habríais': '0',
    'habríamos': '0',
    'habrían': '0',
    'habrías': '0',
    'habéis': '0',
    'había': '0',
    'habíais': '0',
    'habíamos': '0',
    'habían': '0',
    'habías': '0',
    'hace': '0',
    'haceis': '0',
    'hacemos': '0',
    'hacen': '0',
    'hacer': '0',
    'hacerlo': '0',
    'haces': '0',
    'hacia': '0',
    'haciendo': '0',
    'hago': '0',
    'han': '0',
    'has': '0',
    'hasta': '0',
    'hay': '0',
    'haya': '0',
    'hayamos': '0',
    'hayan': '0',
    'hayas': '0',
    'hayáis': '0',
    'he': '0',
    'hecho': '0',
    'hemos': '0',
    'hicieron': '0',
    'hizo': '0',
    'horas': '0',
    'hoy': '0',
    'hube': '0',
    'hubiera': '0',
    'hubierais': '0',
    'hubieran': '0',
    'hubieras': '0',
    'hubieron': '0',
    'hubiese': '0',
    'hubieseis': '0',
    'hubiesen': '0',
    'hubieses': '0',
    'hubimos': '0',
    'hubiste': '0',
    'hubisteis': '0',
    'hubiéramos': '0',
    'hubiésemos': '0',
    'hubo': '0',
    'i': '0',
    'igual': '0',
    'incluso': '0',
    'indicó': '0',
    'informo': '0',
    'informó': '0',
    'intenta': '0',
    'intentais': '0',
    'intentamos': '0',
    'intentan': '0',
    'intentar': '0',
    'intentas': '0',
    'intento': '0',
    'ir': '0',
    'j': '0',
    'junto': '0',
    'k': '0',
    'l': '0',
    'la': '0',
    'lado': '0',
    'largo': '0',
    'las': '0',
    'le': '0',
    'lejos': '0',
    'les': '0',
    'llegó': '0',
    'lleva': '0',
    'llevar': '0',
    'lo': '0',
    'los': '0',
    'luego': '0',
    'lugar': '0',
    'm': '0',
    'mal': '0',
    'manera': '0',
    'manifestó': '0',
    'mas': '0',
    'mayor': '0',
    'me': '0',
    'mediante': '0',
    'medio': '0',
    'mejor': '0',
    'mencionó': '0',
    'menos': '0',
    'menudo': '0',
    'mi': '0',
    'mia': '0',
    'mias': '0',
    'mientras': '0',
    'mio': '0',
    'mios': '0',
    'mis': '0',
    'misma': '0',
    'mismas': '0',
    'mismo': '0',
    'mismos': '0',
    'modo': '0',
    'momento': '0',
    'mucha': '0',
    'muchas': '0',
    'mucho': '0',
    'muchos': '0',
    'muy': '0',
    'más': '0',
    'mí': '0',
    'mía': '0',
    'mías': '0',
    'mío': '0',
    'míos': '0',
    'n': '0',
    'nada': '0',
    'nadie': '0',
    'ni': '0',
    'ninguna': '0',
    'ningunas': '0',
    'ninguno': '0',
    'ningunos': '0',
    'ningún': '0',
    'no': '0',
    'nos': '0',
    'nosotras': '0',
    'nosotros': '0',
    'nuestra': '0',
    'nuestras': '0',
    'nuestro': '0',
    'nuestros': '0',
    'nueva': '0',
    'nuevas': '0',
    'nuevo': '0',
    'nuevos': '0',
    'nunca': '0',
    'o': '0',
    'ocho': '0',
    'os': '0',
    'otra': '0',
    'otras': '0',
    'otro': '0',
    'otros': '0',
    'p': '0',
    'pais': '0',
    'para': '0',
    'parece': '0',
    'parte': '0',
    'partir': '0',
    'pasada': '0',
    'pasado': '0',
    'paìs': '0',
    'peor': '0',
    'pero': '0',
    'pesar': '0',
    'poca': '0',
    'pocas': '0',
    'poco': '0',
    'pocos': '0',
    'podeis': '0',
    'podemos': '0',
    'poder': '0',
    'podria': '0',
    'podriais': '0',
    'podriamos': '0',
    'podrian': '0',
    'podrias': '0',
    'podrá': '0',
    'podrán': '0',
    'podría': '0',
    'podrían': '0',
    'poner': '0',
    'por': '0',
    'por qué': '0',
    'porque': '0',
    'posible': '0',
    'primer': '0',
    'primera': '0',
    'primero': '0',
    'primeros': '0',
    'principalmente': '0',
    'pronto': '0',
    'propia': '0',
    'propias': '0',
    'propio': '0',
    'propios': '0',
    'proximo': '0',
    'próximo': '0',
    'próximos': '0',
    'pudo': '0',
    'pueda': '0',
    'puede': '0',
    'pueden': '0',
    'puedo': '0',
    'pues': '0',
    'q': '0',
    'qeu': '0',
    'que': '0',
    'quedó': '0',
    'queremos': '0',
    'quien': '0',
    'quienes': '0',
    'quiere': '0',
    'quiza': '0',
    'quizas': '0',
    'quizá': '0',
    'quizás': '0',
    'quién': '0',
    'quiénes': '0',
    'qué': '0',
    'r': '0',
    'raras': '0',
    'realizado': '0',
    'realizar': '0',
    'realizó': '0',
    'repente': '0',
    'respecto': '0',
    's': '0',
    'sabe': '0',
    'sabeis': '0',
    'sabemos': '0',
    'saben': '0',
    'saber': '0',
    'sabes': '0',
    'sal': '0',
    'salvo': '0',
    'se': '0',
    'sea': '0',
    'seamos': '0',
    'sean': '0',
    'seas': '0',
    'segun': '0',
    'segunda': '0',
    'segundo': '0',
    'según': '0',
    'seis': '0',
    'ser': '0',
    'sera': '0',
    'seremos': '0',
    'será': '0',
    'serán': '0',
    'serás': '0',
    'seré': '0',
    'seréis': '0',
    'sería': '0',
    'seríais': '0',
    'seríamos': '0',
    'serían': '0',
    'serías': '0',
    'seáis': '0',
    'señaló': '0',
    'si': '0',
    'sido': '0',
    'siempre': '0',
    'siendo': '0',
    'siete': '0',
    'sigue': '0',
    'siguiente': '0',
    'sin': '0',
    'sino': '0',
    'sobre': '0',
    'sois': '0',
    'sola': '0',
    'solamente': '0',
    'solas': '0',
    'solo': '0',
    'solos': '0',
    'somos': '0',
    'son': '0',
    'soy': '0',
    'soyos': '0',
    'su': '0',
    'supuesto': '0',
    'sus': '0',
    'suya': '0',
    'suyas': '0',
    'suyo': '0',
    'suyos': '0',
    'sé': '0',
    'sí': '0',
    'sólo': '0',
    't': '0',
    'tal': '0',
    'tambien': '0',
    'también': '0',
    'tampoco': '0',
    'tan': '0',
    'tanto': '0',
    'tarde': '0',
    'te': '0',
    'temprano': '0',
    'tendremos': '0',
    'tendrá': '0',
    'tendrán': '0',
    'tendrás': '0',
    'tendré': '0',
    'tendréis': '0',
    'tendría': '0',
    'tendríais': '0',
    'tendríamos': '0',
    'tendrían': '0',
    'tendrías': '0',
    'tened': '0',
    'teneis': '0',
    'tenemos': '0',
    'tener': '0',
    'tenga': '0',
    'tengamos': '0',
    'tengan': '0',
    'tengas': '0',
    'tengo': '0',
    'tengáis': '0',
    'tenida': '0',
    'tenidas': '0',
    'tenido': '0',
    'tenidos': '0',
    'teniendo': '0',
    'tenéis': '0',
    'tenía': '0',
    'teníais': '0',
    'teníamos': '0',
    'tenían': '0',
    'tenías': '0',
    'tercera': '0',
    'ti': '0',
    'tiempo': '0',
    'tiene': '0',
    'tienen': '0',
    'tienes': '0',
    'toda': '0',
    'todas': '0',
    'todavia': '0',
    'todavía': '0',
    'todo': '0',
    'todos': '0',
    'total': '0',
    'trabaja': '0',
    'trabajais': '0',
    'trabajamos': '0',
    'trabajan': '0',
    'trabajar': '0',
    'trabajas': '0',
    'trabajo': '0',
    'tras': '0',
    'trata': '0',
    'través': '0',
    'tres': '0',
    'tu': '0',
    'tus': '0',
    'tuve': '0',
    'tuviera': '0',
    'tuvierais': '0',
    'tuvieran': '0',
    'tuvieras': '0',
    'tuvieron': '0',
    'tuviese': '0',
    'tuvieseis': '0',
    'tuviesen': '0',
    'tuvieses': '0',
    'tuvimos': '0',
    'tuviste': '0',
    'tuvisteis': '0',
    'tuviéramos': '0',
    'tuviésemos': '0',
    'tuvo': '0',
    'tuya': '0',
    'tuyas': '0',
    'tuyo': '0',
    'tuyos': '0',
    'tú': '0',
    'u': '0',
    'ultimo': '0',
    'un': '0',
    'una': '0',
    'unas': '0',
    'uno': '0',
    'unos': '0',
    'usa': '0',
    'usais': '0',
    'usamos': '0',
    'usan': '0',
    'usar': '0',
    'usas': '0',
    'uso': '0',
    'usted': '0',
    'ustedes': '0',
    'v': '0',
    'va': '0',
    'vais': '0',
    'valor': '0',
    'vamos': '0',
    'van': '0',
    'varias': '0',
    'varios': '0',
    'vaya': '0',
    'veces': '0',
    'ver': '0',
    'verdad': '0',
    'verdadera': '0',
    'verdadero': '0',
    'vez': '0',
    'vosotras': '0',
    'vosotros': '0',
    'voy': '0',
    'vuestra': '0',
    'vuestras': '0',
    'vuestro': '0',
    'vuestros': '0',
    'w': '0',
    'x': '0',
    'y': '0',
    'ya': '0',
    'yo': '0',
    'z': '0',
    'él': '0',
    'éramos': '0',
    'ésa': '0',
    'ésas': '0',
    'ése': '0',
    'ésos': '0',
    'ésta': '0',
    'éstas': '0',
    'éste': '0',
    'éstos': '0',
    'última': '0',
    'últimas': '0',
    'último': '0',
    'últimos':'0'}
    if text in stop_words:
        return None
    else:
        return text

def conditional_lowercase(word):
    if word.isupper():
        return word.lower()
    else:
        return word

def clean_tweet(tweet):
    #print('tweet cleaned')
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape(""""#$%&'()*+,/:;<=>@[\]^_`{|}~"""), '', tweet)  # remove punctuations (removed . ? - ! to check if it improves)
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    tweet = re.sub('\n', ' ', tweet)
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

    # to check how data is being processed
    with open('corpusss.csv', 'w', newline='', encoding="utf-8") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(texts_list)

    list_tokenized_texts = []

    for i in range(len(texts_list)):
        word_list = text_to_word_sequence(texts_list[i], filters='', lower=False, split=' ')
        word_list = list(map(remove_stop_words, word_list))
        word_list = [elem for elem in word_list if elem]
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
