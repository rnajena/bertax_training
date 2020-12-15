from preprocessing.generate_data import DataSplit
from preprocessing.process_inputs import words2index, words2onehot
from keras.layers import Dense, Embedding
from keras.models import Input, Model
from keras import callbacks as keras_cbs

from tcn import TCN

model_name = 'tcn1'
tcn_options = {'kernel_size': 6,
               'dilations': [2 ** i for i in range(9)],
               'nb_filters': 32}

classes = ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']
nr_seqs = 10_000
batch_size = 500
enc_method_str = 'words2index'
enc_method = {'words2index': words2index, 'words2onehot':
              words2onehot}[enc_method_str]
enc_dimension = 65
emb_layer_dim = 1
max_seq_len = 1000
epochs = 10
split = DataSplit('../../sequences/dna_sequences/', nr_seqs,
                  classes,
                  '../../sequences/dna_sequences/files.json')
train_g, val_g, test_g = split.to_generators(batch_size, enc_method,
                                             enc_dimension,
                                             max_seq_len=max_seq_len,
                                             cache=True)
min_delta = 0.01
patience = 3
early_stopping = keras_cbs.EarlyStopping(min_delta=min_delta,
                                         patience=patience,
                                         restore_best_weights=True)
callbacks = [early_stopping]
# callbacks = []
callbacks.append(keras_cbs.TensorBoard(
    log_dir=f'./logs/{model_name}'))

inputs = Input((max_seq_len,))
emb = Embedding(enc_dimension, emb_layer_dim)(inputs)

o = TCN(return_sequences=False, **tcn_options)(emb)  # The TCN layers are here.
outputs = Dense(len(classes), activation='softmax')(o)
m = Model(inputs=inputs, outputs=outputs)
m.summary()
m.compile(optimizer='adam', loss='categorical_crossentropy',
          metrics=['accuracy'])
hist = m.fit_generator(train_g,
                       validation_data=val_g,
                       epochs=epochs, callbacks=callbacks)

loss, acc = m.evaluate_generator(test_g)
print(f'test loss / test accuracy = {loss:.4f} / {acc:.4f}')
tested = {'loss': loss, 'acc': acc}
