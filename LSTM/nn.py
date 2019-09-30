import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.layers import Dense, Embedding, GRU
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
#from tensorflow.logging import ERROR
#rom tensorflow.logging import set_verbosity

import logging


from sklearn.model_selection import GridSearchCV

import Parser

max_features = 35569
vector_size = 300
WORD_EMBEDDINGS_FILENAME = '../word_embedding/intropln2019_embeddings_es_300.txt'
DATA_TRAIN = '../corpus/data_train.csv'
DATA_TEST = '../corpus/data_test.csv'
DATA_VAL = '../corpus/data_val.csv'

weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
activations = ['elu', 'softmax', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

def create_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=40,
                        weights=[embedding_matrix], trainable=False))
    model.add(GRU(vector_size, dropout=0.2, recurrent_dropout=0.2, kernel_initializer='glorot_uniform',
                   activation='softsign'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def configure_model(model, x_val, y_val):   
    param_grid = dict(weight_constraint=weight_constraint, dropout_rate=dropout_rate,activation=activations, batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum= momentum)
    print(param_grid)
    import pdb; pdb.set_trace
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_val, y_val)
    print(grid_result.best_params_)

    return model

if __name__ == '__main__':
    # Set tensorflow verbosity
    #set_verbosity(ERROR)


    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    #logging.getLogger('tensorflow').setLevel(logging.FATAL)
    #logging.disable(logging.WARNING)
    #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    #deprecation._PRINT_DEPRECATION_WARNINGS = False

    #try:
    #    from tensorflow.python.util import module_wrapper as deprecation
    #except ImportError:
    #    from tensorflow.python.util import deprecation_wrapper as deprecation
    #    deprecation._PER_MODULE_WARNING_LIMIT = 0


    embeddings_index = Parser.embeddings_index(WORD_EMBEDDINGS_FILENAME)
    word_index = Parser.word_index(embeddings_index)
    embedding_matrix = Parser.embedding_matrix(embeddings_index, word_index, vector_size)

    
    #model = KerasClassifier(build_fn=create_model(embedding_matrix), epochs=100, batch_size=10, verbose=0)
    x_val, y_val, _ = Parser.parse_corpus(DATA_VAL, word_index)
    print(x_val)
    import pdb; pdb.set_trace()
    model = configure_model(model, x_val, y_val)

    print('model configured')
    x_train, y_train, _ = Parser.parse_corpus(DATA_TRAIN, word_index)
    batch_size = 32
    epochs = 4
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, workers=2, use_multiprocessing=True)

    #x_test, _y_test, _ = Parser.parse_corpus(DATA_TETS, word_index)
    #loss, metrics = model.evaluate(x_test, _y_test, batch_size=batch_size, workers=2, use_multiprocessing=True)
    #print('Loss: ', loss)
    #print('Accuracy: ', metrics)
