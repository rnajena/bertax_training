import keras
import json
from preprocessing.process_inputs import get_class_vectors, ALPHABET
from models.model import PARAMS
import numpy as np
from keras.utils import Sequence
from models.bert_utils import get_token_dict, seq2tokens, predict
from models.bert_utils import generate_bert_with_pretrained
from random import shuffle, sample
from sklearn.model_selection import train_test_split
import os.path
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from logging import warning
import pickle


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


@dataclass
class FragmentGenerator(Sequence):
    x: list
    y: list
    seq_len: int
    max_seq_len: Optional[int] = None
    k: int = 3
    stride: int = 3
    batch_size: int = 32
    classes: List = field(default_factory=lambda:
                          ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'])
    seq_len_like: Optional[np.array] = None
    window: bool = False

    def __post_init__(self):
        self.class_vectors = get_class_vectors(self.classes)
        self.token_dict = get_token_dict(ALPHABET, k=3)
        if (self.max_seq_len is None):
            self.max_seq_len = self.seq_len

    def __len__(self):
        return np.ceil(len(self.x)
                       / float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        batch_fragments = self.x[idx * self.batch_size:
                                 (idx + 1) * self.batch_size]
        batch_x = [seq2tokens(seq, self.token_dict, seq_length=self.seq_len,
                              max_length=self.max_seq_len,
                              k=self.k, stride=self.stride, window=self.window,
                              seq_len_like=self.seq_len_like)
                   for seq in batch_fragments]
        if (self.y is not None and len(self.y) != 0):
            batch_classes = self.y[idx * self.batch_size:
                                   (idx + 1) * self.batch_size]
            batch_y = np.array([self.class_vectors[c] for c in batch_classes])
            return ([np.array([_[0] for _ in batch_x]),
                     np.array([_[1] for _ in batch_x])], [batch_y])
        else:
            return [np.array([_[0] for _ in batch_x]),
                    np.array([_[1] for _ in batch_x])]


def get_fine_model(pretrained_model_file):
    model_fine = generate_bert_with_pretrained(
        pretrained_model_file, len(classes))
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = model_fine.input_shape[0][1]
    return model_fine, max_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='fine-tune BERT on pre-generated fragments')
    parser.add_argument('pretrained_bert')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--seq_len_like', default=None,
                        help='path of pickled class dict of seq lens for '
                        'generating sampled sequence sizes')
    parser.add_argument('--k', help=' ', default=3, type=int)
    parser.add_argument('--stride', help=' ', default=3, type=int)
    parser.add_argument('--batch_size', help=' ', type=int, default=32)
    parser.add_argument('--epochs', help=' ', type=int, default=4)
    parser.add_argument('--nr_seqs', help=' ', type=int, default=260_000)
    parser.add_argument('--learning_rate', help=' ', type=float, default=5e-5)
    parser.add_argument('--save_name',
                        help='custom name for saved finetuned model',
                        default=None)
    parser.add_argument('--store_predictions', help=' ', action='store_true')
    parser.add_argument('--roc_auc', help=' ', action='store_true')
    args = parser.parse_args()
    learning_rate = args.learning_rate
    if (args.seq_len_like is not None):
        seq_len_dict = pickle.load(open(args.seq_len_like, 'rb'))
        min_nr_seqs = min(map(len, seq_len_dict.values()))
        seq_len_like = []
        for k in seq_len_dict:
            seq_len_like.extend(np.random.choice(seq_len_dict[k], min_nr_seqs)
                                // args.k)
    else:
        seq_len_like = None
    # building model
    model, max_length = get_fine_model(args.pretrained_bert)
    if (args.seq_len > max_length):
        warning(f'desired seq len ({args.seq_len}) is higher than possible ({max_length})'
                f'setting seq len to {max_length}')
        args.seq_len = max_length
    generator_args = {
        'max_seq_len': max_length, 'k': args.k, 'stride': args.stride,
        'batch_size': args.batch_size, 'window': True,
        'seq_len_like': seq_len_like}
    model.summary()
    # loading training data
    x, y = load_fragments(args.fragments_dir, nr_seqs=args.nr_seqs)
    f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
        x, y, test_size=0.2)
    f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
        f_train_x, f_train_y, test_size=0.05)
    model.fit(
        FragmentGenerator(f_train_x, f_train_y, args.seq_len,
                          **generator_args),
        epochs=args.epochs,
        validation_data=FragmentGenerator(f_val_x, f_val_y, args.seq_len,
                                          **generator_args))
    if (args.save_name is not None):
        save_path = args.save_name + '.h5'
    else:
        save_path = os.path.splitext(args.pretrained_bert)[0] + '_finetuned.h5'
    model.save(save_path)
    print('testing...')
    test_g = FragmentGenerator(f_test_x, f_test_y, args.seq_len,
                               **generator_args)
    if (args.store_predictions or args.roc_auc):
        predicted = predict(
            model, test_g,
            args.roc_auc, classes, return_data=args.store_predictions)
        result = predicted['metrics']
        metrics_names = predicted['metrics_names']
        if (args.store_predictions):
            import pickle
            pickle.dump(predicted, open(os.path.splitext(save_path)[0]
                                        + '_predictions.pkl', 'wb'))
    else:
        result = model.evaluate(test_g)
        metrics_names = model.metrics_names
    print("test results:", *zip(metrics_names, result))
