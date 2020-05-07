import keras
import json
from preprocessing.process_inputs import seq2kmers, get_class_vectors, ALPHABET
from models.model import PARAMS
import numpy as np
from keras.utils import Sequence
from models.bert_utils import get_token_dict, seq2tokens, generate_bert_with_pretrained
from random import shuffle, sample
from sklearn.model_selection import train_test_split
import os.path
import argparse


classes = PARAMS['data']['classes'][1]


def load_fragments(fragments_dir, shuffle_=True, balance=True, nr_seqs=None):
    fragments = []
    for class_ in classes:
        fragments.append((class_, json.load(open(os.path.join(
            fragments_dir, f'{class_}_fragments.json')))))
    nr_seqs_max = min(len(_[1]) for _ in fragments)
    if (nr_seqs is None or nr_seqs > nr_seqs_max):
        nr_seqs = nr_seqs_max
    x = []
    y = []
    for class_, class_fragments in fragments:
        if not balance:
            x.extend(class_fragments)
            y.extend([class_] * len(class_fragments))
        else:
            x.extend(sample(class_fragments, nr_seqs))
            y.extend([class_] * nr_seqs)
    assert len(x) == len(y)
    if (shuffle_):
        to_shuffle = list(zip(x, y))
        shuffle(to_shuffle)
        x, y = zip(*to_shuffle)
    print(f'{len(x)} fragments loaded in total; '
          f'balanced={balance}, shuffle_={shuffle_}, nr_seqs={nr_seqs}')
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
        return np.ceil(len(self.x)
                       / float(batch_size)).astype(np.int)

    def __getitem__(self, idx):
        global token_dict
        batch_fragments = self.x[idx * batch_size:
                                 (idx + 1) * batch_size]
        batch_classes = self.y[idx * batch_size:
                               (idx + 1) * batch_size]
        batch_y = np.array([self.class_vectors[c] for c in batch_classes])
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False, to_upper=True)
                      for seq in batch_fragments]
        batch_x = []
        for seq in batch_seqs:
            batch_x.append(seq2tokens(seq, token_dict, self.seq_len,
                                      window=False))
        return ([np.array([_[0] for _ in batch_x]),
                np.array([_[1] for _ in batch_x])], batch_y)


def get_fine_model(pretrained_model_file):
    model_fine, max_length = generate_bert_with_pretrained(
        pretrained_model_file, len(classes))
    model_fine.summary()
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    return model_fine, max_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='fine-tune BERT on pre-generated fragments')
    parser.add_argument('pretrained_bert')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--batch_size', help=' ', type=int, default=32)
    parser.add_argument('--epochs', help=' ', type=int, default=4)
    parser.add_argument('--nr_seqs', help=' ', type=int, default=250_000)
    parser.add_argument('--learning_rate', help=' ', type=float, default=5e-5)
    args = parser.parse_args()
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
    model.fit(
        FragmentGenerator(f_train_x, f_train_y, max_length),
        epochs=args.epochs,
        validation_data=FragmentGenerator(f_val_x, f_val_y, max_length))
    model.save(os.path.splitext(args.pretrained_bert)[0] + '_finetuned.h5')
    print('testing...')
    result = model.evaluate(FragmentGenerator(f_test_x, f_test_y, max_length))
    print(result)
