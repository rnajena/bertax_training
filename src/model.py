import numpy as np
import tensorflow as tf
if (tf.__version__.startswith('1.')):
    import keras
else:
    import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Conv1D, Dropout, MaxPooling1D, Input
from keras.layers import Embedding, Dense, Flatten
from keras.layers import concatenate, LSTM, Bidirectional, GRU
from keras.utils import plot_model, Sequence
from keras import optimizers
import keras.callbacks as keras_cbs
from dataclasses import dataclass
from generate_data import DataSplit
from process_inputs import kmers2index, kmers2onehot
from logging import info, warning
from datetime import datetime
import json
from glob import glob
import re


def model_name(model=None, unique=True):
    """returns hours:minutes.  If unique is specified, '_x' will be
    appended so that the name will be unique among all models (with logs)"""
    basename = datetime.now().strftime('%H-%M')
    path = f'logs/{basename}'
    existing = [e for e in glob(path + '*')
                if re.match(path + r'_?[0-9]*', e)]
    existing = sorted(existing, key=lambda x:
                      int(x.split('_')[-1])
                      if '_' in x else 0)
    if (len(existing) == 1 and existing[0] == path):
        existing[0] += '_0'
    last_nr = (int(existing[-1].split('_')[-1]) if len(existing) != 0
               else None)
    return basename + (f'_{last_nr + 1}' if last_nr is not None else '')


# {group: {param_name:
#   (type|(list, type), default, model_type, help, choices)}}
PARAMS = {'nns': {'emb_layer_dim': (int, 1),
                  'dropout_rate': (float, 0.01),
                  'max_pool': (bool, True, 'cnn'),
                  'summary': (bool, True),
                  'plot': (bool, False),
                  # cnn
                  'nr_filters': (int, 16, 'cnn,tcn'),
                  'kernel_size': (int, 4, 'cnn,tcn'),
                  'nr_layers': (int, 16, 'cnn'),
                  'neurons_full': (int, 64, 'cnn'),
                  'conv_strides': (int, 1, 'cnn'),
                  # lstm
                  'lstm_units': (int, 32, 'lstm'),
                  'cell_type': (str, 'lstm', 'lstm', '',
                                ['lstm', 'gru']),
                  'bidirectional': (bool, True, 'lstm'),
                  # tcn
                  'last_dilation_2exp': (int, 9, 'tcn',
                                         'base-2 exponent of last dilation; '
                                         'eg.: 3: dilations=[1,2,4,8]')},
          'data': {'classes':
                   ((list, str),
                    ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']),
                   'nr_seqs': (int, 10_000), 'batch_size': (int, 500),
                   'rev_comp': (bool, False), 'rev_comp_mode': (
                       str, 'append', None, '', ['append', 'random']),
                   'enc_dimension': (int, 65),
                   'cache_batches': (bool, True),
                   'cache_seq_limit': (int, None),
                   'root_fa_dir':
                   (str, '/home/lo63tor/master/sequences/dna_sequences/'),
                   'file_names_cache':
                   (str,
                    '/home/lo63tor/master/sequences/dna_sequences/files.json'),
                   'enc_method': (str, 'kmers2index', None, '',
                                  ['kmers2index', 'kmers2onehot']),
                   'max_seq_len': (int, 10_000), },
          'run': {'epochs': (int, 100), 'test_split': (float, 0.2),
                  'model_name': (str, model_name()),
                  'class_report': (bool, True, None,
                                   'prints metrics by class for evalutation'),
                  'early_stopping': (bool, True),
                  'early_stopping_md': (float, 0.01, None, 'min_delta'),
                  'early_stopping_p': (int, 5, None, 'patience'),
                  'early_stopping_restore_weights':
                  (bool, True, None, 'restore best weights')}, }


@dataclass
class DCModel:
    """initializes a keras model for DNA classficication.

    max_seq_len will be adapted to actual maximum length of the encoded
    sequences"""
    classes: list
    max_seq_len: int = 100
    enc_dimension: int = 64
    name: str = model_name()

    def __post_init__(self):
        self.trained = None
        self.tested = None

    # functions for building models with various architectures.
    # set the `model` instance variable

    def generate_cnn_model(self, emb_layer_dim=None, nr_filters=8,
                           kernel_size=16, nr_layers=2, neurons_full=32,
                           conv_strides=1, dropout_rate=0.3, max_pool=True,
                           summary=True, plot=False):
        info('generating model')
        input_shape = ((self.max_seq_len, self.enc_dimension)
                       if emb_layer_dim is None else (self.max_seq_len,))
        inputs = Input(shape=input_shape)
        if (emb_layer_dim is not None):
            emb = Embedding(self.enc_dimension, emb_layer_dim)(inputs)
            inputs_to_connect = emb
        else:
            inputs_to_connect = inputs
        stack = []
        for i in range(nr_layers):
            if type(nr_filters) == list:
                nr_filters = nr_filters[i]
            if type(kernel_size) == list:
                kernel_size = kernel_size[i]
            conv = Conv1D(filters=nr_filters, kernel_size=kernel_size,
                          strides=conv_strides, activation='relu')(
                              inputs_to_connect)
            if (dropout_rate is not None and dropout_rate > 0):
                dropout = Dropout(dropout_rate)(conv)
            else:
                dropout = conv
            if (max_pool):
                pool = MaxPooling1D(nr_filters)(dropout)
            else:
                pool = dropout
            # flatten = Flatten()(pool)
            flatten = Flatten()(pool)
            stack.append(flatten)
        merged = concatenate(stack)
        # conv2 = Conv1D(filters=nr_filters, kernel_size=kernel_size,strides=1,
        #                activation='relu')(stack)
        # flatten = Flatten()(conv2)
        full_con = Dense(neurons_full, activation='relu')(merged)
        outputs = Dense(len(self.classes), activation='softmax')(full_con)
        model = Model(inputs=inputs, outputs=outputs)
        if (summary):
            model.summary()
        if (plot):
            plot_model(model, 'model.png')
        self.model = model

    def generate_lstm_model(self,
                            cell_type='lstm',
                            bidirectional=False,
                            emb_layer_dim=1,
                            lstm_units=32,
                            dropout_rate=0.3,
                            summary=True, plot=False):
        # adapted from https://keras.io/examples/imdb_bidirectional_lstm/
        input_shape = (self.max_seq_len,)
        inputs = Input(shape=input_shape)
        emb = Embedding(self.enc_dimension, emb_layer_dim)(inputs)
        cell_type = {'lstm': LSTM, 'gru': GRU}[cell_type.lower()]
        if (bidirectional):
            rec_layer = Bidirectional(cell_type(lstm_units))(emb)
        else:
            rec_layer = cell_type(lstm_units)(emb)
        if (dropout_rate is not None and dropout_rate > 0):
            dropout = Dropout(dropout_rate)(rec_layer)
        else:
            dropout = rec_layer
        outputs = Dense(len(self.classes), activation='softmax')(dropout)
        model = Model(inputs=inputs, outputs=outputs)
        if (summary):
            model.summary()
        if (plot):
            plot_model(model, 'model.png')
        self.model = model

    def generate_tcn_model(self,
                           emb_layer_dim=1,
                           kernel_size=6, dilations=[2 ** i for i in range(9)],
                           nb_filters=32, dropout_rate=0.0,
                           summary=True, plot=False):
        # NOTE: only works with tensorflow 1.*
        from tcn import TCN
        input_shape = (self.max_seq_len,)
        inputs = Input(shape=input_shape)
        emb = Embedding(self.enc_dimension, emb_layer_dim)(inputs)
        o = TCN(return_sequences=False,
                kernel_size=kernel_size, dilations=dilations,
                nb_filters=nb_filters, dropout_rate=dropout_rate)(emb)
        outputs = Dense(len(self.classes), activation='softmax')(o)
        model = Model(inputs=inputs, outputs=outputs)
        if (summary):
            model.summary()
        if (plot):
            plot_model(model, 'model.png')
        self.model = model

    def generate_ff_model(self, summary=True, plot=False):
        input_shape = (self.max_seq_len, self.enc_dimension)
        inputs = Input(shape=input_shape)
        full_con2 = Dense(500, activation='relu')(inputs)
        full_con3 = Dense(50, activation='relu')(full_con2)
        flatten = Flatten()(full_con3)
        outputs = Dense(len(self.classes), activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        if (summary):
            model.summary()
        if (plot):
            plot_model(model, 'model.png')
        self.model = model

    def train(self, train_generator: Sequence,
              val_generator: Sequence = None,
              optimizer='adam', batch_size=None, epochs=100,
              val_split=0.05, tensorboard=True, callbacks=[]):
        info(f'training with {epochs} epochs')
        if (optimizer == 'adam'):
            optimizer = optimizers.Adam(learning_rate=0.003)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        if (tensorboard):
            callbacks.append(keras_cbs.TensorBoard(
                log_dir=f'./logs/{self.name}'))
        # self.model.fit(self.data[0], self.data[2], batch_size=32,
        #                epochs=epochs, validation_split=val_split,
        #                callbacks=callbacks)
        hist = self.model.fit_generator(train_generator,
                                        validation_data=val_generator,
                                        epochs=epochs, callbacks=callbacks)
        self.trained = hist
        return hist

    def eval(self, test_generator: Sequence, class_report=True):
        # loss, acc = self.model.evaluate(self.data[1], self.data[3])
        loss, acc = self.model.evaluate_generator(test_generator)
        print(f'test loss / test accuracy = {loss:.4f} / {acc:.4f}')
        if class_report:
            from sklearn.metrics import classification_report
            y_pred = self.model.predict_generator(test_generator)
            y_pred = np.argmax(y_pred, 1)
            y_true = []
            for i in range(len(test_generator)):
                y_true.extend(test_generator[i][1])
            y_true = np.argmax(y_true, 1)
            c_report = classification_report(y_true, y_pred,
                                             target_names=self.classes)
            print(c_report)
        self.tested = {'loss': loss, 'acc': acc,
                       'classification_report': c_report}
        return acc

    def write_to_file(self, filename, params={}):
        results = {}
        if (self.trained is not None):
            results['history'] = {key: [float(v) for v in val] for key, val in
                                  self.trained.history.items()}
        if (self.tested is not None):
            results['test_metrics'] = {key: (float(val) if type(val) != str
                                             else val)
                                       for key, val in self.tested.items()}
        if (len(params) != 0):
            results['params'] = params
        if (len(results) == 0):
            warning('model doesn\'t seem to have been trained, '
                    'not writing to file')
            return
        results['name'] = self.name
        with open(filename, 'w') as f:
            json.dump(results, f)


def cnn(epochs=200, early_stopping=True,
        classes=['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
        batch_size=5000, max_seq_len=10_000, nr_seqs=1000, enc_dimension=65,
        emb_layer_dim=1, enc_method=kmers2index, model_settings={}):
    m = DCModel(classes, max_seq_len=max_seq_len, enc_dimension=enc_dimension)
    m.generate_cnn_model(emb_layer_dim=emb_layer_dim, **model_settings)
    split = DataSplit('../../sequences/dna_sequences/', nr_seqs,
                      classes,
                      '../../sequences/dna_sequences/files.json')
    train_g, val_g, test_g = split.to_generators(batch_size, enc_method,
                                                 enc_dimension,
                                                 max_seq_len=max_seq_len)
    callbacks = []
    if (early_stopping):
        early_stopping = keras_cbs.EarlyStopping(min_delta=0.001, patience=30,
                                                 restore_best_weights=True)
        callbacks.append(early_stopping)
    m.train(train_g, val_g, epochs=epochs, callbacks=callbacks)
    m.eval(test_g)
    m.write_to_file('model_results_test.json')


def blstm(epochs=50, early_stopping=True,
          classes=['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
          batch_size=250, max_seq_len=10_000, nr_seqs=25_000, enc_dimension=65,
          emb_layer_dim=1, enc_method=kmers2index, model_settings={}):
    m = DCModel(classes, max_seq_len=max_seq_len, enc_dimension=enc_dimension)
    m.generate_cnn_model(emb_layer_dim=emb_layer_dim, **model_settings)
    split = DataSplit('../../sequences/dna_sequences/', nr_seqs,
                      classes,
                      '../../sequences/dna_sequences/files.json')
    train_g, val_g, test_g = split.to_generators(batch_size, enc_method,
                                                 enc_dimension,
                                                 max_seq_len=max_seq_len,
                                                 cache=True)
    early_stopping = keras_cbs.EarlyStopping(min_delta=0.01, patience=5,
                                             restore_best_weights=True)
    callbacks = [early_stopping]
    m.train(train_g, val_g, epochs=epochs, callbacks=callbacks)
    return m.eval(test_g)


def onehot_cnn():
    cnn(epochs=100, batch_size=1000, enc_dimension=64, emb_layer_dim=None,
        enc_method=kmers2onehot)


def emb_cnn():
    cnn(epochs=200, early_stopping=True,
        classes=['Viruses', 'Archaea'],
        nr_seqs=100_000, batch_size=5_000,
        enc_dimension=65, emb_layer_dim=1,
        enc_method=kmers2index)


def grid_search():
    classes = ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']
    nr_seqs = 100_000
    batch_size = 500
    enc_method_str = 'kmers2index'
    enc_method = {'kmers2index': kmers2index, 'kmers2onehot':
                  kmers2onehot}[enc_method_str]
    enc_dimension = 65
    emb_layer_dim = 1
    max_seq_len = 10_000
    epochs = 100
    split = DataSplit('../../sequences/dna_sequences/', nr_seqs,
                      classes,
                      '../../sequences/dna_sequences/files.json')
    train_g, val_g, test_g = split.to_generators(batch_size, enc_method,
                                                 enc_dimension,
                                                 max_seq_len=max_seq_len,
                                                 cache=True)
    # early stopping adapted to low batch_size/high #batches
    # seems to converge after ~3 epochs
    min_delta = 0.01
    patience = 3
    early_stopping = keras_cbs.EarlyStopping(min_delta=min_delta,
                                             patience=patience,
                                             restore_best_weights=True)
    callbacks = [early_stopping]
    defaults = {'nr_filters': 8, 'kernel_size': 4,
                'nr_layers': 16, 'neurons_full': 64,
                'dropout_rate': 0.01, 'emb_layer_dim': 1}
    grid = {'kernel_size': [12],
            'nr_layers': [32], 'emb_layer_dim': [2, 4]}
    keys = list(grid.keys())
    accuracies = {}
    for key in keys:
        for value in grid[key]:
            model_settings = {k: (defaults[k] if k != key else value)
                              for k in defaults.keys()}
            print(model_settings)
            m = DCModel(classes, max_seq_len=max_seq_len,
                        enc_dimension=enc_dimension)
            m.name = f'grid2_{key}_{value}'
            m.generate_cnn_model(**model_settings)
            m.train(train_g, val_g, epochs=epochs, callbacks=callbacks)
            params = {}
            params.update(defaults)
            params.update(grid)
            params.update({
                'classes': classes,
                'nr_seqs': nr_seqs,
                'batch_size': batch_size,
                'enc_method': enc_method_str,
                'enc_dimension': enc_dimension,
                'emb_layer_dim': emb_layer_dim,
                'max_seq_len': max_seq_len,
                'epochs': epochs,
                'min_delta': min_delta,
                'patience': patience})
            accuracies[(key, value)] = m.eval(test_g)
            m.write_to_file(str(id(m)) + m.name + '.json', params)
    return accuracies


if __name__ == '__main__':
    exit(0)
    # NOTE: nr_seqs !<= 239_756
    # NOTE: batch_size ~ 5000 -> 28GB RAM
    # NOTE: batch_size ~ 20_000 -> 83GB with cache
    # NOTE: batches+cache: 250 batches with 100k sequences -> 12GB RAM
    # emb_cnn()
    print(grid_search())
    # callbacks
    # callbacks = []
    # reduce_lr = ReduceLROnPlateau()
    # callbacks.append(reduce_lr)
    # early_stopping = keras_cbs.EarlyStopping(min_delta=0.01, patience=5)
    # callbacks.append(early_stopping)
    # m.train(train_g, val_g, epochs=100, callbacks=callbacks)
    # m.eval(test_g)
    # DEBUG
    # split = DataSplit('../../sequences/dna_sequences/', 10,
    #                   classes,
    #                   '../../sequences/dna_sequences/files.json')
    # train_g, val_g, test_g = split.to_generators(batch_size, kmers2index, 65,
    #                                              max_seq_len=max_seq_len)
