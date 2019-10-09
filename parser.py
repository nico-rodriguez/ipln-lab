#!/usr/bin/env python3

import pandas as pd


def load_data(data_filename):
    print('Leyendo archivo {filename}...'.format(filename=data_filename))
    df = pd.read_csv(data_filename)
    return df['text'].tolist(), df['humor'].tolist()


def preprocess_data(texts_list):
    print('Pre procesando textos...')
    return ['hola', 'hola2', 'hola3']
