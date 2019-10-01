from keras.callbacks import CSVLogger
from keras.layers import Bidirectional, average, Dense, Embedding, GRU, Input, LSTM
from keras.models import Model, Sequential
from keras.utils import plot_model
from tensorflow.logging import ERROR
from tensorflow.logging import set_verbosity
from matplotlib import pyplot
from Metrics import Metrics
import Parser


def single_lstm(embedding_matrix):
    units = 30
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=True))
    model.add(LSTM(units=units, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def triple_lstm(embedding_matrix):
    units = 30
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=True))
    model.add(LSTM(units=units, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign', return_sequences=True))
    model.add(LSTM(units=units, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign', return_sequences=True))
    model.add(LSTM(units=units, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def bidirectional_lstm(embedding_matrix):
    units = 30
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=True))
    model.add(Bidirectional(LSTM(units=units//2, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                                 return_sequences=True, activation='softsign')))
    model.add(Bidirectional(LSTM(units=units//2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_gru(embedding_matrix):
    units = 30
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=True))
    model.add(LSTM(units=units//2, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign', return_sequences=True))
    model.add(GRU(units=units//2, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                  activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def parallel_lstm_gru(embedding_matrix):
    units = 30
    inputs = Input(shape=(40,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                          weights=[embedding_matrix], trainable=True)(inputs)
    lstm = LSTM(units=units//2, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                activation='softsign')(embedding)
    output1 = Dense(1, activation='softsign')(lstm)
    gru = GRU(units=units//2, dropout=0.2, recurrent_dropout=0.2)(embedding)
    output2 = Dense(1, activation='relu')(gru)
    output = average([output1, output2])
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def parallel_bidirectional_lstm_gru(embedding_matrix):
    units = 30
    inputs = Input(shape=(40,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                          weights=[embedding_matrix], trainable=True)(inputs)
    bi_lstm_1 = Bidirectional(LSTM(units=units//3, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                                   return_sequences=True, activation='softsign'))(embedding)
    bi_lstm_2 = Bidirectional(LSTM(units=units//3))(bi_lstm_1)
    output1 = Dense(1, activation='softsign')(bi_lstm_2)
    gru = GRU(units=units//3, dropout=0.2, recurrent_dropout=0.2)(embedding)
    output2 = Dense(1, activation='relu')(gru)
    output = average([output1, output2])
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def parallel_bidirectional_lstm_lstm_gru(embedding_matrix):
    units = 30
    inputs = Input(shape=(40,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                          weights=[embedding_matrix], trainable=True)(inputs)
    bi_lstm_1 = Bidirectional(LSTM(units=units//4, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                                   return_sequences=True, activation='softsign'))(embedding)
    bi_lstm_2 = Bidirectional(LSTM(units=units//4))(bi_lstm_1)
    output1 = Dense(1, activation='softsign')(bi_lstm_2)
    lstm = LSTM(units=units//4, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                activation='softsign')(embedding)
    output2 = Dense(1, activation='tanh')(lstm)
    gru = GRU(units=units//4, dropout=0.2, recurrent_dropout=0.2)(embedding)
    output3 = Dense(1, activation='relu')(gru)
    output = average([output1, output2, output3])
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_model(model_name, model, x_train, y_train, x_val, y_val):
    #plot_model(model, to_file=model_name+'_model.png', expand_nested=True)
    batch_size = 32
    epochs = 10
    csv_logger = CSVLogger(model_name+'_training.log')
    metrics = Metrics()
    metrics.set_file_name(model_name+'_metrics.log')
    print('epoch,val_f1,val_prec,val_rec', file=open(model_name+'_metrics.log', 'w'))
    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[metrics, csv_logger],
                              workers=2, use_multiprocessing=True, validation_freq=1, validation_data=(x_val, y_val))
    # pyplot training history
    '''
    pyplot.plot(train_history.history['loss'], label='train loss')
    pyplot.plot(train_history.history['acc'], label='train acc')
    pyplot.plot(train_history.history['val_loss'], label='val loss')
    pyplot.plot(train_history.history['val_acc'], label='val acc')
    pyplot.legend()
    pyplot.savefig(model_name + '_training_history.png', bbox_inches='tight')
    pyplot.close()
    # pyplot metrics history
    pyplot.plot(metrics.val_f1s, label='val_f1')
    pyplot.plot(metrics.val_precisions, label='val_prec')
    pyplot.plot(metrics.val_recalls, label='val_rec')
    pyplot.legend()
    pyplot.savefig(model_name + '_metrics_history.png', bbox_inches='tight')
    pyplot.close()
    '''



if __name__ == '__main__':
    #../word_embedding/intropln2019_embeddings_es_300.txt
    #../word_embedding/SBW-vectors-300-min5.txt
    WORD_EMBEDDINGS_FILENAME = '../word_embedding/intropln2019_embeddings_es_300.txt'
    DATA_TRAIN = '../corpus/data_train.csv'
    DATA_VAL = '../corpus/data_val.csv'

    # Set tensorflow verbosity
    set_verbosity(ERROR)

    max_features = 35569
    vector_size = 300
    #remove_unknown_words = True
    remove_unknown_words = False
    perform_clean_up = False

    embeddings_index = Parser.embeddings_index(WORD_EMBEDDINGS_FILENAME)
    word_index = Parser.word_index(embeddings_index, remove_unknown_words)
    embedding_matrix = Parser.embedding_matrix(embeddings_index, word_index, vector_size, remove_unknown_words)

    x_train, y_train, _ = Parser.parse_corpus(DATA_TRAIN, word_index, max_features, remove_unknown_words, perform_clean_up)
    x_val, y_val, _ = Parser.parse_corpus(DATA_VAL, word_index, max_features, remove_unknown_words, perform_clean_up)
    
    
    model = lstm_gru(embedding_matrix)
    model.summary()
    test_model('lstm_gru', model, x_train, y_train, x_val, y_val)
    
    '''
    model = single_lstm(embedding_matrix)
    test_model('single_lstm', model, x_train, y_train, x_val, y_val)
    model = bidirectional_lstm(embedding_matrix)
    test_model('bidirectional_lstm', model, x_train, y_train, x_val, y_val)
    model = lstm_gru(embedding_matrix)
    test_model('lstm_gru', model, x_train, y_train, x_val, y_val)
    model = parallel_lstm_gru(embedding_matrix)
    test_model('parallel_lstm_gru', model, x_train, y_train, x_val, y_val)
    model = parallel_bidirectional_lstm_gru(embedding_matrix)
    test_model('parallel_bidirectional_lstm_gru', model, x_train, y_train, x_val, y_val)
    model = parallel_bidirectional_lstm_lstm_gru(embedding_matrix)
    test_model('parallel_bidirectional_lstm_lstm_gru', model, x_train, y_train, x_val, y_val)
    '''