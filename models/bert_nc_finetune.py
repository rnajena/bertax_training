if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# import tensorflow.keras as keras
import keras
import keras_bert
import json
from preprocessing.process_inputs import seq2kmers, get_class_vectors, ALPHABET
from models.model import PARAMS
import numpy as np
from keras.utils import Sequence
from models.bert_nc import get_token_dict
from random import shuffle
from sklearn.model_selection import train_test_split
import os.path
from argparse import ArgumentParser


classes = PARAMS['data']['classes'][1]


def load_fragments(fragments_dir, shuffle_=True):
    x = []
    y = []
    for class_ in classes:
        for fragment in json.load(open(os.path.join(
            fragments_dir, f'{class_}_fragments.json'))):
         x.append(fragment)
         y.append(class_)
    if (shuffle_):
        to_shuffle = list(zip(x, y))
        shuffle(to_shuffle)
        x, y = zip(*to_shuffle)
    return x, y


class FragmentGenerator(Sequence):

    def __init__(self, x, y, seq_len):
        global classes
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.class_vectors = get_class_vectors(classes)

    def __len__(self):
        global batch_size
        return np.ceil(len(self.x) /
                       float(batch_size)).astype(np.int)

    def __getitem__(self, idx):
        global token_dict
        batch_fragments = self.x[idx * batch_size:
                                 (idx+1) * batch_size]
        batch_classes = self.y[idx * batch_size:
                              (idx+1) * batch_size]
        batch_y = np.array([self.class_vectors[c] for c in batch_classes])
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False, to_upper=True)
                      for seq in batch_fragments]
        batch_x = []
        for seq in batch_seqs:
            indices = [token_dict['[CLS]']] + [token_dict[word]
                                               if word in token_dict
                                               else token_dict['[UNK]']
                                               for word in seq]
            if (len(indices) < self.seq_len):
                indices += [token_dict['']] * (self.seq_len - len(indices))
            else:
                indices = indices[:self.seq_len]
            segments = [0 for _ in range(self.seq_len)]
            batch_x.append([np.array(indices), np.array(segments)])
        return ([np.array([_[0] for _ in batch_x]),
                np.array([_[1] for _ in batch_x])], batch_y)

def get_fine_model(pretrained_model_file):
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(pretrained_model_file, compile=False,
                                    custom_objects=custom_objects)
    # construct fine-tuning model as in
    # https://colab.research.google.com/github/CyberZHG/keras-bert/blob/master/demo/tune/keras_bert_classification_tpu.ipynb
    inputs = model.inputs[:2]
    nsp_dense_layer = model.get_layer(name='NSP-Dense').output
    softmax_layer = keras.layers.Dense(4, activation='softmax')(nsp_dense_layer)
    model_fine = keras.Model(inputs=inputs, outputs=softmax_layer)
    model_fine.summary()
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = inputs[0].shape[1]
    return model_fine, max_length


def process_bert_tokens_batch(batch_x):
    return [np.array([_[0] for _ in batch_x]),
            np.array([_[1] for _ in batch_x])]

if __name__ == '__main__':
    parser = ArgumentParser(
        description='fine-tune BERT on pre-generated fragments')
    parser.add_argument('pretrained_bert')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', type=int, default=502)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--nr_seqs', type=int, default=250_000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    args = parser.parse_args()
    # args = parser.parse_args(['output/bert_nc_ep13.h5',
    #                           '/home/lo63tor/master/dna_class/output/genomic_fragments/',
    #                           '--seq_len', '502', '--batch_size', '32',
    #                           '--nr_seqs', '1000'])
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    token_dict = get_token_dict(ALPHABET, k=3)
    # building model
    model, max_length = get_fine_model(args.pretrained_bert)
    model.summary()
    # loading training data
    x, y = load_fragments(args.fragments_dir)
    f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
        x, y, test_size=0.2)
    f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
        f_train_x, f_train_y, test_size=0.05)
    # val_g = FragmentGenerator(f_val_x, f_val_y, max_length)
    model.fit(
        FragmentGenerator(f_train_x, f_train_y, max_length),
        epochs=args.epochs,
        validation_data=FragmentGenerator(f_val_x, f_val_y, max_length))
    model.save('bert_nc_finetuned.h5')
    result = model.evaluate(FragmentGenerator(f_test_x, f_test_y, max_length))
    print(result)
