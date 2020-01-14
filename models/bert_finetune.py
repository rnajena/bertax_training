# import tensorflow.keras as keras
import keras
import keras_bert
import json
from preprocessing.process_inputs import seq2kmers
from preprocessing.generate_data import DataSplit
from models.model import PARAMS
from random import randint
import numpy as np
import sys
exit()

# bert paper recommendations:
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4
# Batch size: 16, 32
learning_rate = 5e-5
epochs = 5
batch_size = 32
nr_seqs = 250_000

root_fa_dir = sys.argv[1]
file_names_cache = sys.argv[2]
# root_fa_dir = PARAMS['data']['root_fa_dir'][1]
classes = PARAMS['data']['classes'][1]
# file_names_cache = PARAMS['data']['file_names_cache'][1]

custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                  'GlorotUniform': keras.initializers.glorot_uniform}
custom_objects.update(keras_bert.get_custom_objects())
model = keras.models.load_model('bert.h5', compile=False,
                                custom_objects=custom_objects)
# keras.utils.plot_model(model, 'bert_v0.png')

# construct fine-tuning model as in
# https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/tune/keras_bert_classification_tpu.ipynb
inputs = model.inputs[:2]
nsp_dense_layer = model.get_layer(name='NSP-Dense').output
# extract_layer = model.get_layer(name='Extract')
softmax_layer = keras.layers.Dense(4, activation='softmax')(nsp_dense_layer)
model_fine = keras.Model(inputs=inputs, outputs=softmax_layer)
model_fine.summary()
model_fine.compile(keras.optimizers.Adam(learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# load token_dict from pre-trained model
token_dict = json.load(open('keras-bert_token_dict.json'))

# DataGenerator
max_length = inputs[0].shape[1]


def seq2tokens(seq, window=True):
    seq = seq2kmers(seq, k=3, stride=3, pad=True)
    if (window):
        start = randint(0, max(len(seq) - max_length - 1, 0))
        end = start + max_length - 1
    else:
        start = 0
        end = len(seq)
    indices = [token_dict['[CLS]']] + [token_dict[word] for word in
                                       seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]


def process_batch(batch_x):
    return [np.array([_[0] for _ in batch_x]),
            np.array([_[1] for _ in batch_x])]


split = DataSplit(root_fa_dir=root_fa_dir, nr_seqs=nr_seqs, classes=classes,
                  from_cache=file_names_cache)
train_g, val_g, test_g = split.to_generators(
    batch_size=batch_size, custom_encode_sequence=seq2tokens,
    process_batch_function=process_batch)
model_fine.fit_generator(train_g, validation_data=val_g,
                         epochs=epochs)
result = model_fine.evaluate_generator(test_g)
print(result)
