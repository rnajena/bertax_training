import tensorflow as tf
from tensorflow import keras
import json
from preprocessing.process_inputs import get_class_vectors, ALPHABET
from models.model import PARAMS
import numpy as np
from tensorflow.keras.utils import Sequence
from models.bert_utils import get_token_dict, seq2tokens, predict
from models.bert_utils import generate_bert_with_pretrained, generate_bert_with_pretrained_multi_tax
from random import shuffle, sample
from sklearn.model_selection import train_test_split
import os.path
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from logging import warning
import pickle
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from os.path import splitext
import pandas as pd

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
mirrored_strategy = tf.distribute.MirroredStrategy()

classes = PARAMS['data']['classes'][1]


def load_dataset(filepath):
    df = pd.read_csv(filepath, sep="\t")
    x = df["x"]
    y = df["y"]
    y_species = df["tax_id"]
    return x, y, y_species


def load_fragments(fragments_dir, shuffle_=True, balance=True, nr_seqs=None):
    fragments = []
    species_list = []
    for class_ in classes:
        fragments.append((class_, json.load(open(os.path.join(
            fragments_dir, f'{class_}_fragments.json')))))
        species_list.append([int(line.strip()) for line in
                             open(os.path.join(fragments_dir, f'{class_}_species_picked.txt')).readlines()])
    nr_seqs_max = min(len(_[1]) for _ in fragments)
    if (nr_seqs is None or nr_seqs > nr_seqs_max):
        nr_seqs = nr_seqs_max
    x = []
    y = np.array([])
    y_species = np.array([], dtype=np.int)

    for index, fragments_i in enumerate(fragments):
        class_, class_fragments = fragments_i
        if not balance:
            x.extend(class_fragments)
            y = np.append(y, [class_] * len(class_fragments))
            y_species = np.append(y_species, species_list[index])
            gc.collect()
        else:
            x_help = list(zip(class_fragments, species_list[index]))
            # x.extend(sample(class_fragments, nr_seqs))
            x_help = sample(x_help, nr_seqs)
            x_help, y_species_help = zip(*x_help)
            x.extend(x_help)
            y_species.extend(y_species_help)
            y.extend([class_] * nr_seqs)

    assert len(x) == len(y)
    if (shuffle_):
        to_shuffle = list(zip(x, y, y_species))
        shuffle(to_shuffle)
        x, y, y_species = zip(*to_shuffle)
    print(f'{len(x)} fragments loaded in total; '
          f'balanced={balance}, shuffle_={shuffle_}, nr_seqs={nr_seqs}')
    return np.array(x), np.array(y), np.array(y_species)


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


@dataclass
class FragmentGenerator_multi_tax(Sequence):
    x: list
    y: list
    y_species: list
    weight_classes: dict
    classes: dict
    seq_len: int
    tax_ranks: list
    max_seq_len: Optional[int] = None
    k: int = 3
    stride: int = 3
    batch_size: int = 32
    seq_len_like: Optional[np.array] = None
    window: bool = False

    def get_class_vectors_multi_tax(self, taxid):
        vector = []
        weight = []

        ranks = self.tlineage.get_ranks(taxid, ranks=self.tax_ranks)

        # calc vector per tax and append
        # if class not in dict use vector of unknown class
        for index, class_tax_i in enumerate(ranks):
            vector.append(self.class_vectors[self.tax_ranks[index]].get(ranks[class_tax_i][1],
                                                                        self.class_vectors[self.tax_ranks[index]][
                                                                            'unknown']))

        # calc sample weight
        for index, class_tax_i in enumerate(ranks):
            weight.append(self.weight_classes[self.tax_ranks[index]].get(ranks[class_tax_i][1],
                                                                         self.weight_classes[self.tax_ranks[index]][
                                                                             'unknown']))

        return vector, weight

    def __post_init__(self):
        # from utils.tax_entry import TaxDB
        # self.taxDB = TaxDB(data_dir="/mnt/fass2/projects/fm_read_classification_comparison/taxonomy")
        from utils.tax_entry import TaxidLineage
        tlineage = TaxidLineage()
        self.tlineage = tlineage
        self.class_vectors = dict()
        for tax_rank in self.classes:
            self.class_vectors.update({tax_rank: get_class_vectors(self.classes[tax_rank])})

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
            batch_classes = self.y_species[idx * self.batch_size:
                                           (idx + 1) * self.batch_size]
            batch_y, weights = zip(*[self.get_class_vectors_multi_tax(taxid) for taxid in batch_classes])
            # batch_y, weights = zip(*[self.get_class_vectors_multi_tax(taxid) for taxid in batch_classes])
            X = [np.array([_[0] for _ in batch_x]), np.array([_[1] for _ in batch_x])]
            y = []
            weights_y = []
            for index in range(len(self.tax_ranks)):
                y.append(np.array([_[index] for _ in batch_y]))
                weights_y.append(np.array([_[index] for _ in weights]))
            return (X, y, weights_y)
        else:
            return [np.array([_[0] for _ in batch_x]),
                    np.array([_[1] for _ in batch_x])]


def get_classes_and_weights_multi_tax(species_list, tax_ranks=['superkingdom', 'kingdom', 'family'],
                                      unknown_thr=10_000):
    from utils.tax_entry import TaxidLineage
    tlineage = TaxidLineage()

    classes = dict()
    weight_classes = dict()
    tax_ranks_dict = dict()
    num_entries = len(species_list)
    species_list_y = []
    for tax_rank_i in tax_ranks:
        tax_ranks_dict.update({tax_rank_i: dict()})

    for taxid in species_list:
        ranks = tlineage.get_ranks(taxid, ranks=tax_ranks)
        taxid_y = []
        for tax_rank_i in tax_ranks:
            num_same_tax_rank_i = tax_ranks_dict[tax_rank_i].get(ranks[tax_rank_i][1], 0) + 1
            tax_ranks_dict[tax_rank_i].update({ranks[tax_rank_i][1]: num_same_tax_rank_i})
            taxid_y.append(ranks[tax_rank_i][1])
        species_list_y.append(taxid_y)

    for index, key in enumerate(tax_ranks_dict.keys()):
        dict_ = tax_ranks_dict[key]
        classes_tax_i = dict_.copy()
        unknown = 0
        weight_classes_tax_i = dict()
        for key, value in dict_.items():
            if value < unknown_thr:
                unknown += value
                classes_tax_i.pop(key)
            else:
                weight = num_entries / value
                weight_classes_tax_i.update({key: weight})

        unknown += classes_tax_i.get("unknown", 0)
        classes_tax_i.update({'unknown': unknown})
        classes.update({tax_ranks[index]: classes_tax_i})

        weight = num_entries / unknown if unknown != 0 else 1
        weight_classes_tax_i.update({'unknown': weight})
        weight_classes.update({tax_ranks[index]: weight_classes_tax_i})

    species_list_y = np.array(species_list_y)
    species_list_y = np.array([i if i in classes[tax_ranks[j]] else 'unknown' for j in range(len(tax_ranks)) for i in
                               species_list_y[:, j]]).reshape((len(tax_ranks), -1)).swapaxes(0, 1)

    return classes, weight_classes, species_list_y


def get_fine_model(pretrained_model_file):
    # with mirrored_strategy.scope():
    model_fine = generate_bert_with_pretrained(
        pretrained_model_file, len(classes))
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = model_fine.input_shape[0][1]
    return model_fine, max_length


def get_fine_model_multi_tax(pretrained_model_file, num_classes):
    # with mirrored_strategy.scope():
    model_fine = generate_bert_with_pretrained_multi_tax(
        pretrained_model_file, num_classes)
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = model_fine.input_shape[0][1]
    tf.keras.utils.plot_model(model_fine, to_file="model.png", show_shapes=True)
    model_fine.summary()

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
    parser.add_argument('--store_train_data', help=' ', action='store_true')
    parser.add_argument('--roc_auc', help=' ', action='store_true')
    parser.add_argument('--multi_tax', help=' ', action='store_true')
    parser.add_argument('--test_benchmark', help=' ', action='store_true')

    args = parser.parse_args()

    tax_ranks = ["superkingdom", "phylum"]
    print("main")

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

    # model, max_length = get_fine_model_multi_tax(args.pretrained_bert, num_classes=(2,17,22))
    # exit()
    if not args.test_benchmark:
        # loading training data
        x, y, y_species = load_fragments(args.fragments_dir, nr_seqs=args.nr_seqs)

        if args.multi_tax:
            x_help = list(zip(x, y_species))
            f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
                x_help, y, test_size=0.2, stratify=y)
            f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
                f_train_x, f_train_y, test_size=0.05, stratify=f_train_y)
            f_train_x, f_train_y_species = zip(*f_train_x)
            f_val_x, f_val_y_species = zip(*f_val_x)
            f_test_x, f_test_y_species = zip(*f_test_x)

            classes, weight_classes, species_list_y = get_classes_and_weights_multi_tax(f_train_y_species,
                                                                                        tax_ranks=tax_ranks)

        else:
            f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
                x, y, test_size=0.2, stratify=y)
            f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
                f_train_x, f_train_y, test_size=0.05, stratify=f_train_y)

    else:
        f_test_x, f_test_y, f_test_y_species = load_dataset("/home/go96bix/projects/dna_class/resources/test.tsv")
        f_train_x, f_train_y, f_train_y_species = load_dataset("/home/go96bix/projects/dna_class/resources/train.tsv")
        f_train_x = list(zip(f_train_x, f_train_y_species))
        f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(f_train_x, f_train_y, test_size=0.05,
                                                                  stratify=f_train_y)
        f_train_x, f_train_y_species = zip(*f_train_x)
        f_val_x, f_val_y_species = zip(*f_val_x)
        classes, weight_classes, species_list_y = get_classes_and_weights_multi_tax(f_train_y_species,
                                                                                    tax_ranks=tax_ranks)

    # building model
    if args.multi_tax:
        num_classes = (len(classes[tax].keys()) for tax in classes)
        model, max_length = get_fine_model_multi_tax(args.pretrained_bert, num_classes=num_classes)
    else:
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

    filepath1 = splitext(args.pretrained_bert)[0] + "model.best.acc.hdf5"
    filepath2 = splitext(args.pretrained_bert)[0] + "model.best.loss.hdf5"
    checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='max')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='min')
    checkpoint3 = EarlyStopping('val_accuracy', min_delta=0, patience=2, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir="./logs/run_repeated_undersampling", histogram_freq=1, write_graph=True,
                                       write_images=True, update_freq=100, embeddings_freq=1)

    # callbacks_list = [checkpoint1, checkpoint2, checkpoint3]
    callbacks_list = [checkpoint1, checkpoint2, checkpoint3, tensorboard_callback]

    if (args.store_train_data):
        from datetime import datetime

        time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
        for kind, x, y in [('train', f_train_x, f_train_y), ('val', f_val_x, f_val_y),
                           ('test', f_test_x, f_test_y)]:
            with open(f'{time_str}_{kind}_data.json', 'w') as f:
                json.dump([x, y], f)
        print('saved train/test/val data.')
    if args.multi_tax:
        model.fit(
            FragmentGenerator_multi_tax(f_train_x, f_train_y, f_train_y_species, weight_classes, seq_len=args.seq_len,
                                        tax_ranks=tax_ranks, classes=classes, **generator_args),
            callbacks=callbacks_list, epochs=args.epochs,
            validation_data=FragmentGenerator_multi_tax(f_val_x, f_val_y, f_val_y_species, weight_classes,
                                                        seq_len=args.seq_len, tax_ranks=tax_ranks, classes=classes,
                                                        **generator_args), use_multiprocessing=False, workers=0)
    else:
        model.fit(FragmentGenerator(f_train_x, f_train_y, args.seq_len, **generator_args), callbacks=callbacks_list,
                  epochs=args.epochs,
                  validation_data=FragmentGenerator(f_val_x, f_val_y, args.seq_len, **generator_args))
    if (args.save_name is not None):
        save_path = args.save_name + '.h5'
    else:
        save_path = os.path.splitext(args.pretrained_bert)[0] + '_finetuned.h5'
    model.save(save_path)
    print('testing...')
    if args.multi_tax:
        test_g = FragmentGenerator_multi_tax(f_test_x, f_test_y, f_test_y_species, weight_classes, seq_len=args.seq_len,
                                             classes=classes, **generator_args)
    else:
        test_g = FragmentGenerator(f_test_x, f_test_y, args.seq_len, **generator_args)
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
