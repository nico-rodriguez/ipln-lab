from keras.layers import Dense, Embedding, LSTM, GRU
from keras.models import Sequential
from tensorflow.logging import ERROR
from tensorflow.logging import set_verbosity

import Parser

if __name__ == '__main__':
    WORD_EMBEDDINGS_FILENAME = '../word_embedding/intropln2019_embeddings_es_300.txt'
    DATA_TRAIN = '../corpus/data_train.csv'
    DATA_TETS = '../corpus/data_test.csv'

    # Set tensorflow verbosity
    set_verbosity(ERROR)

    max_features = 35569
    vector_size = 300

    embeddings_index = Parser.embeddings_index(WORD_EMBEDDINGS_FILENAME)
    word_index = Parser.word_index(embeddings_index)
    embedding_matrix = Parser.embedding_matrix(embeddings_index, word_index, vector_size)

    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=False))
    model.add(LSTM(vector_size, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign', return_sequences=True))
    model.add(GRU(units=vector_size))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train, y_train, _ = Parser.parse_corpus(DATA_TRAIN, word_index)
    batch_size = 32
    epochs = 4
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, workers=2, use_multiprocessing=True)

    x_test, _y_test, _ = Parser.parse_corpus(DATA_TETS, word_index)
    loss, metrics = model.evaluate(x_test, _y_test, batch_size=batch_size, workers=2, use_multiprocessing=True)
    print('Loss: ', loss)
    print('Accuracy: ', metrics)
