import numpy as np
from dataclasses import dataclass
from preprocessing.generate_data import DataSplit
from preprocessing.process_inputs import words2index, words2onehot, words2vec
from logging import info, warning
from datetime import datetime
import json
from glob import glob
import re
import tensorflow as tf
if (tf.__version__.startswith('1.')):
    from keras.models import Model, load_model
    from keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D, Input
    from keras.layers import Embedding, Dense, Flatten
    from keras.layers import concatenate, LSTM, Bidirectional, GRU
    from keras.utils import plot_model, Sequence
    from keras import optimizers
    import keras.callbacks as keras_cbs
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D
    from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
    from tensorflow.keras.layers import concatenate, LSTM, Bidirectional, GRU
    from tensorflow.keras.utils import plot_model, Sequence
    from tensorflow.keras import optimizers
    import tensorflow.keras.callbacks as keras_cbs


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
PARAMS = {'nns':
          # * hyper-parameters of neural networks *
          {'emb_layer_dim': (int, 1),
           'dropout_rate': (float, 0.01),
           'max_pool': (bool, True, 'cnn'),
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
                                  'eg.: 3: dilations=[1,2,4,8]'),
           'noncausal_dilations': (bool, False, 'tcn')},
          'data':
          # * everything data-related *
          {'data_source':
           (str, 'genes', None,
            'dataset type to use', ['genes', 'fragments']),
           'classes':
           ((list, str),
            ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']),
           'nr_seqs': (int, 10_000), 'batch_size': (int, 500),
           'fixed_size_method': (
               str, 'pad', None,
               'Method for transforming sequences to fixed length',
               ['pad', 'window', 'repeat']),
           'rev_comp': (bool, False), 'rev_comp_mode': (
               str, 'append', None, '', ['append', 'random',
                                         'independent']),
           'enc_dimension': (int, 65),
           'enc_k': (int, 3),
           'enc_stride': (int, 3),
           'cache_batches': (bool, True),
           'cache_seq_limit': (int, None),
           'root_fa_dir':
           (str, '/home/lo63tor/master/sequences/dna_sequences/'),
           'root_fragments_dir':
           (str, '/home/lo63tor/master/dna_class/output/genomic_fragments'),
           'file_names_cache':
           (str,
            '/home/lo63tor/master/sequences/dna_sequences/files.json'),
           'enc_method':
           (str, 'words2index', None, '',
            ['words2index', 'words2onehot', 'words2vec']),
           'w2vfile': (str, None, None, 'filename of a pickled word '
                       'vector dict'),
           'bert_token_dict_json':
           (str, '', None, 'path to the JSON-serialized keras-bert '
            'token dict'),
           'bert_pretrained_path':
           (str, '', None, 'path to pre-trained keras-bert model'),
           'max_seq_len': (int, 10_000, None,
                           'Length of *all* sequences when '
                           'using any `fixed_size_method`')},
          'run':
          # ** run/training-related **
          {'epochs': (int, 100), 'learning_rate': (float, 0.003),
           'test_split': (float, 0.2),
           'model_name': (str, model_name()),
           'class_report': (bool, True, None,
                            'prints metrics by class for evalutation'),
           'early_stopping': (bool, True),
           'early_stopping_md': (float, 0.01, None, 'min_delta'),
           'early_stopping_p': (int, 5, None, 'patience'),
           'early_stopping_restore_weights':
           (bool, True, None, 'restore best weights'),
           'model_checkpoints': (bool, False, None,
                                 'saves model after every epoch'),
           'model_checkpoints_keep_all': (bool, False, None,
                                          'checkpoints won\'t be overridden'),
           'summary': (bool, False),
           'plot': (bool, False),
           'save':
           (bool, False, None, 'save model to `model_name`.h5')}}


@dataclass
class DCModel:
    """initializes a keras model for DNA classficication.

    max_seq_len will be adapted to actual maximum length of the encoded
    sequences"""
    classes: list
    max_seq_len: int = 100
    enc_dimension: int = 64
    name: str = model_name()
    summary: bool = False
    plot: bool = False
    save: bool = False

    def __post_init__(self):
        self.trained = None
        self.tested = None

    # functions for building models with various architectures.
    # set the `model` instance variable

    def _model_inputs(self, emb_layer_dim=None):
        if (emb_layer_dim == 0):
            emb_layer_dim = None
        input_shape = ((self.max_seq_len, self.enc_dimension)
                       if emb_layer_dim is None else (self.max_seq_len,))
        inputs = Input(shape=input_shape, name='input')
        if (emb_layer_dim is None):
            return inputs, inputs
        emb = Embedding(self.enc_dimension, emb_layer_dim,
                        name='embedding')(inputs)
        return inputs, emb

    def _model_outputs(self, last_layer):
        return Dense(len(self.classes), activation='softmax',
                     name='classification')(last_layer)

    def _model_visualization(self):
        if (self.summary):
            self.model.summary()
        if (self.plot):
            plot_model(self.model, 'model.png')
        if (self.save):
            self.model.save(self.name + '.h5')

    def generate_cnn_model(self, emb_layer_dim=None, nr_filters=8,
                           kernel_size=16, nr_layers=2, neurons_full=32,
                           conv_strides=1, dropout_rate=0.3, max_pool=True):
        info('generating model')
        inputs, emb = self._model_inputs(emb_layer_dim)
        stack = []
        for i in range(nr_layers):
            if type(nr_filters) == list:
                nr_filters = nr_filters[i]
            if type(kernel_size) == list:
                kernel_size = kernel_size[i]
            conv = Conv1D(filters=nr_filters, kernel_size=kernel_size,
                          strides=conv_strides, activation='relu')(
                              emb)
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
        outputs = self._model_outputs(full_con)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def generate_cnndeep_model(self, emb_layer_dim=None, nr_filters=8,
                               kernel_size=16, nr_layers=2,
                               neurons_full=32, conv_strides=1,
                               dropout_rate=0.3, max_pool=True):
        info('generating model')
        inputs, emb = self._model_inputs(emb_layer_dim)
        # layer 1
        conv = Conv1D(filters=nr_filters, kernel_size=kernel_size,
                      strides=conv_strides, activation='relu')(
                          emb)
        pool = MaxPooling1D(4)(conv)
        # layer 2
        conv2 = Conv2D(filters=nr_filters, kernel_size=kernel_size,
                       activation='relu')(pool)
        pool2 = MaxPooling1D(8)(conv2)
        drop = Dropout(dropout_rate)(pool2)
        flatten = Flatten()(drop)
        full_con = Dense(neurons_full, activation='relu')(flatten)
        outputs = self._model_outputs(full_con)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def generate_cnndeep_predef_model(self, emb_layer_dim=1, nr_filters=72,
                                      dropout_rate=0.1, max_pool=True,
                                      **ignored_kwargs):
        info('generating model')
        stack_kernel_sizes = [6, 20, 6]
        inputs, emb = self._model_inputs(emb_layer_dim)
        # layer 1
        conv1 = Conv1D(filters=nr_filters, kernel_size=stack_kernel_sizes[0],
                       strides=1, activation='relu')(
                          emb)
        # if (dropout_rate is not None and dropout_rate > 0):
        #     dropout = Dropout(dropout_rate)(conv1)
        # else:
        #     dropout = conv1
        # layer 2
        dropout1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(filters=nr_filters, kernel_size=stack_kernel_sizes[1],
                       activation='relu')(dropout1)
        # flatten = Flatten()(pool)
        # conv2 = Conv1D(filters=nr_filters, kernel_size=kernel_size,strides=1,
        #                activation='relu')(stack)
        # layer 3
        dropout2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(filters=nr_filters, kernel_size=stack_kernel_sizes[2],
                       activation='relu')(dropout2)
        # flatten = Flatten()(pool)
        # conv2 = Conv1D(filters=nr_filters, kernel_size=kernel_size,strides=1,
        #                activation='relu')(stack)
        dropout3 = Dropout(dropout_rate)(conv3)
        flatten1 = Flatten('channels_last')(dropout1)
        flatten2 = Flatten('channels_last')(dropout2)
        flatten3 = Flatten('channels_last')(dropout3)
        merged = concatenate([flatten1, flatten2, flatten3])
        # if (max_pool):
        #     pool = MaxPooling1D(4)(flatten3)
        # else:
        #     pool = flatten3
        full_con = Dense(64, activation='relu')(merged)
        outputs = self._model_outputs(full_con)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def generate_lstm_model(self, emb_layer_dim=None, cell_type='lstm',
                            bidirectional=False, lstm_units=32,
                            dropout_rate=0.3):
        # adapted from https://keras.io/examples/imdb_bidirectional_lstm/
        inputs, emb = self._model_inputs(emb_layer_dim)
        cell_type = {'lstm': LSTM, 'gru': GRU}[cell_type.lower()]
        if (bidirectional):
            rec_layer = Bidirectional(cell_type(lstm_units))(emb)
        else:
            rec_layer = cell_type(lstm_units)(emb)
        if (dropout_rate is not None and dropout_rate > 0):
            dropout = Dropout(dropout_rate)(rec_layer)
        else:
            dropout = rec_layer
        outputs = self._model_outputs(dropout)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def generate_tcn_model(self, emb_layer_dim=None, kernel_size=6,
                           dilations=[2 ** i for i in range(9)],
                           nb_filters=32, dropout_rate=0.0, noncausal=False):
        # NOTE: only works with tensorflow 1.*
        from tcn import TCN
        inputs, emb = self._model_inputs(emb_layer_dim)
        o = TCN(return_sequences=False,
                kernel_size=kernel_size, dilations=dilations,
                nb_filters=nb_filters, dropout_rate=dropout_rate,
                padding=('same' if noncausal else 'causal'))(emb)
        outputs = self._model_outputs(o)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def generate_ff_model(self, emb_layer_dim=None, summary=True, plot=False,
                          **ignored_kwargs):
        inputs, emb = self._model_inputs(emb_layer_dim)
        full_con = Dense(100, activation='sigmoid')(emb)
        flatten = Flatten()(full_con)
        outputs = self._model_outputs(flatten)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._model_visualization()

    def load_model(self, path, custom_objects={}):
        # NOTE: load_model is prone to fail when using custom
        # libraries, a model from a different version, ...
        self.model = load_model(path, custom_objects=custom_objects)
        self._model_visualization()

    def generate_bert_with_pretrained(self, pretrained_path, **ignored_kwargs):
        from models.bert_utils import generate_bert_with_pretrained
        self.model = generate_bert_with_pretrained(pretrained_path,
                                                   len(self.classes))
        self._model_visualization()

    def train(self, train_generator: Sequence,
              val_generator: Sequence = None,
              optimizer='adam', batch_size=None, epochs=100,
              val_split=0.05, tensorboard=False, callbacks=[],
              learning_rate=0.003):
        info(f'training with {epochs} epochs')
        if (optimizer == 'adam'):
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        if (tensorboard):
            callbacks.append(keras_cbs.TensorBoard(
                log_dir=f'./logs/{self.name}'))
        # self.model.fit(self.data[0], self.data[2], batch_size=32,
        #                epochs=epochs, validation_split=val_split,
        #                callbacks=callbacks)
        hist = self.model.fit(train_generator,
                              validation_data=val_generator,
                              epochs=epochs, callbacks=callbacks)
        self.trained = hist
        if (self.save):
            self.model.save(self.name + '.h5')
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

    def write_to_file(self, filename, params={}, specific_params={}):
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
        if (len(specific_params) != 0):
            results['specific_params'] = specific_params
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
        emb_layer_dim=1, enc_method=words2index, model_settings={}):
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
          emb_layer_dim=1, enc_method=words2index, model_settings={}):
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
        enc_method=words2onehot)


def emb_cnn():
    cnn(epochs=200, early_stopping=True,
        classes=['Viruses', 'Archaea'],
        nr_seqs=100_000, batch_size=5_000,
        enc_dimension=65, emb_layer_dim=1,
        enc_method=words2index)


def grid_search():
    classes = ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']
    nr_seqs = 100
    batch_size = 500
    enc_method_str = 'words2index'
    enc_method = {'words2index': words2index, 'words2onehot':
                  words2onehot, 'words2vec': words2vec}[enc_method_str]
    enc_dimension = 100
    emb_layer_dim = 1
    max_seq_len = 10_000
    epochs = 20
    split = DataSplit('../../sequences/dna_sequences/', nr_seqs,
                      classes,
                      '../../sequences/dna_sequences/files.json')
    train_g, val_g, test_g = split.to_generators(batch_size,
                                                 enc_method=enc_method,
                                                 enc_dimension=enc_dimension,
                                                 enc_k=3, enc_stride=3,
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
                'nr_layers': 8, 'neurons_full': 64,
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
            m.name = f'grid_cnn_singlenuleotide_{key}_{value}'
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
    # train_g, val_g, test_g = split.to_generators(batch_size, words2index, 65,
    #                                              max_seq_len=max_seq_len)
