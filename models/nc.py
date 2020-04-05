if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import json
import os.path
from random import shuffle
from sklearn.model_selection import train_test_split
from preprocessing.process_inputs import encode_sequence, ALPHABET, get_class_vectors, words2index
import keras
from models.model import PARAMS, DCModel
from argparse import ArgumentParser
from itertools import product
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from logging import warning

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
        batch_fragments = self.x[idx * batch_size:
                                 (idx+1) * batch_size]
        batch_classes = self.y[idx * batch_size:
                              (idx+1) * batch_size]
        batch_y = np.array([self.class_vectors[c] for c in batch_classes])
        batch_x = [encode_sequence(seq, 'window', words2index, k=3, stride=3,
                                      max_seq_len=self.seq_len)
                      for seq in batch_fragments]
        return batch_x, batch_y

if __name__ == '__main__':
    parser = ArgumentParser(
        description='train cnndeep model on nc fragments')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    # args = parser.parse_args(['/home/lo63tor/master/dna_class/output/genomic_fragments/',
    #                           '--seq_len', '502', '--batch_size', '70', '--head_num', '2',
    #                           '--transformer_num', '2', '--embed_dim', '10', '--epochs', '3'])
    batch_size = args.batch_size
    # building model
    model = DCModel(classes, args.seq_len, 65, summary=True)
    model.generate_cnndeep_predef_model(dropout_rate=0)
    # loading training data
    x, y = load_fragments(args.fragments_dir)
    f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
        x, y, test_size=0.2)
    f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
        f_train_x, f_train_y, test_size=0.05)
    model.train(FragmentGenerator(f_train_x, f_train_y, args.seq_len),
                FragmentGenerator(f_val_x, f_val_y, args.seq_len),
                batch_size=batch_size, epochs=args.epochs)
    model.eval(FragmentGenerator(f_test_x, f_test_y, args.seq_len))
