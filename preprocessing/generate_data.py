from os import scandir
import os.path
from random import sample, shuffle, choice
from preprocessing.process_inputs import words2onehot, words2index, words2vec
from preprocessing.process_inputs import seq2kmers, seq2nucleotides
from preprocessing.process_inputs import encode_sequence, read_seq
from preprocessing.process_inputs import get_class_vectors
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.utils import Sequence
import logging
from logging import info, warning, debug
import json
import pickle
from tqdm import tqdm
from typing import Optional, Callable
from Bio.Seq import Seq
import itertools
import re


@dataclass
class DataSplit:
    """NOTE: splits in favor of train data"""
    # TODO: improve split to not be this inequal
    root_fa_dir: str
    nr_seqs: int
    classes: list
    from_cache: Optional[str] = None
    from_cache_format: str = 'json'
    train_test_split: float = 0.2
    val_split: float = 0.05
    duplicate_data: Optional[str] = None
    balance: bool = True
    shuffle_: bool = True

    def __post_init__(self):
        self.get_fa_files(self.from_cache, self.from_cache_format)
        self.process_fa_files(self.balance, self.shuffle_)
        # files will be split in the order: train, val, test

        def abs_range(length, split, offset=0):
            return (offset, offset+np.ceil(length*split).astype(np.int) - 1)
        self.ranges = {}
        self.ranges['train'] = abs_range(len(self.labels),
                                         (1 - self.train_test_split)
                                         * (1 - self.val_split))
        self.ranges['val'] = abs_range(len(self.labels),
                                       (1 - self.train_test_split)
                                       * self.val_split,
                                       self.ranges['train'][1] + 1)
        self.ranges['test'] = abs_range(len(self.labels),
                                        self.train_test_split,
                                        self.ranges['val'][1] + 1)

    def store_seq_file_names(self, cache_file, mode='json'):
        file_names = []
        labels = []
        for c in self.classes:
            with open(os.path.join(self.root_fa_dir, c, c + '.fa')) as f:
                file_names_c = {
                    os.path.join(c, 'split', line.split()[0].split('|')[-1]
                                 + '.fa')
                    for line in f.readlines() if line.startswith('>')}
                labels_c = [c] * len(file_names_c)
                file_names.extend(list(file_names_c))
                labels.extend(labels_c)
        if (mode == 'json'):
            with open(cache_file, 'w') as f:
                json.dump([file_names, labels], f)
        else:
            with open(cache_file, 'wb') as f:
                pickle.dump([file_names, labels], f)

    def get_fa_files(self, from_cache=None, from_cache_format='json'):
        """reads in file names of all sequence files(*.fa) in
        self.root_fa_dir/{class_dir}/split along with class labels.
        When `from_cache` is specified as a file name, attempts to
        read in a list of file names plus classes from this file
        NOTE: reading file names without cache takes a very long time
        """
        self.file_names = []
        self.labels = []
        if (from_cache is not None):
            if (not os.path.isfile(from_cache)):
                raise Exception(f'{from_cache} is not a valid file name')
            if (from_cache_format == 'json'):
                with open(from_cache) as f:
                    info('reading in cached file names')
                    self.file_names, self.labels = json.load(f)
                    if (self.duplicate_data is not None):
                        info('duplicating file names and labels')
                        self.file_names += [f'{_}${self.duplicate_data}'
                                            for _ in self.file_names]
                        self.labels *= 2
                    self.file_names = [os.path.join(self.root_fa_dir, f)
                                       for f in self.file_names]
            else:
                with open(from_cache, 'rb') as f:
                    info('reading in cached file names')
                    self.file_names, self.labels = pickle.load(f)
        else:
            info('generating list of file names. This will take a while!')
            for c in self.classes:
                class_dir = os.path.join(self.root_fa_dir, c, 'split')
                if (not os.path.isdir(class_dir)):
                    raise Exception(f'class {c}\'s fasta dir doesn\'t exist',
                                    class_dir)
                for fasta in tqdm(scandir(class_dir)):
                    file_name = fasta.path
                    if not file_name.endswith('.fa'):
                        continue
                    self.file_names.append(file_name)
                    self.labels.append(c)

    def process_fa_files(self, balance=True, shuffle_=True):
        """shuffles and balances files"""
        # balance so that every class has {lowest_seq_nr} sequences
        info('sorting sequences by class')
        class_indices = {c: [i for i, label in enumerate(self.labels)
                             if label == c]
                         for c in self.classes}
        lowest_seq_nr = min([len(class_indices[c]) for c in self.classes])
        if (self.nr_seqs == 0):
            self.nr_seqs = lowest_seq_nr
        if (lowest_seq_nr < self.nr_seqs):
            warning('not enough sequences found, using '
                    f'{lowest_seq_nr} instead of {self.nr_seqs} sequences')
            self.nr_seqs = lowest_seq_nr
        file_names_b = []
        labels_b = []
        if (balance):
            info(f'balancing data with {lowest_seq_nr} sequences for each class')
            for c in self.classes:
                for i in sample(class_indices[c], self.nr_seqs):
                    file_names_b.append(self.file_names[i])
                    labels_b.append(self.labels[i])
        else:
            file_names_b = self.file_names
            labels_b = self.labels
        if (not shuffle_):
            return
        # shuffle lists
        info('shuffling data')
        to_shuffle = list(zip(file_names_b, labels_b))
        shuffle(to_shuffle)
        self.file_names, self.labels = zip(*to_shuffle)

    def get_train_files(self):
        data_range = self.ranges['train']
        file_names = self.file_names[data_range[0]: data_range[1] + 1]
        labels = self.labels[data_range[0]: data_range[1] + 1]
        # add remainig unused data
        last_test_index = self.ranges['test'][1]
        if (last_test_index < len(self.labels) - 1):
            file_names.extend(self.file_names[last_test_index + 1:])
            labels.extend(self.labels[last_test_index + 1:])
        return (file_names, labels)

    def get_val_files(self):
        data_range = self.ranges['val']
        return (self.file_names[data_range[0]: data_range[1] + 1],
                self.labels[data_range[0]: data_range[1] + 1])

    def get_test_files(self):
        data_range = self.ranges['test']
        return (self.file_names[data_range[0]: data_range[1] + 1],
                self.labels[data_range[0]: data_range[1] + 1])

    def to_generators(self, batch_size, rev_comp=False, rev_comp_mode='append',
                      fixed_size_method='pad', enc_method=words2onehot,
                      enc_dimension=64, enc_k=3, enc_stride=3,
                      max_seq_len=10_000, force_max_len=True, cache=False,
                      cache_seq_limit=None, w2vfile=None,
                      custom_encode_sequence=None,
                      process_batch_function=None) -> tuple:
        kwargs = {'classes': self.classes, 'batch_size': batch_size,
                  'rev_comp': rev_comp, 'rev_comp_mode': rev_comp_mode,
                  'fixed_size_method': fixed_size_method,
                  'enc_method': enc_method, 'enc_dimension': enc_dimension,
                  'enc_k': enc_k, 'enc_stride': enc_stride,
                  'max_seq_len': max_seq_len, 'force_max_len': force_max_len,
                  'cache': cache, 'cache_seq_limit': cache_seq_limit,
                  'w2vfile': w2vfile,
                  'custom_encode_sequence': custom_encode_sequence,
                  'process_batch_function': process_batch_function}
        info('splitting to train,test and validation data generators')
        split_seqs_nr = {p: self.ranges[p][1] - self.ranges[p][0] + 1
                         for p in self.ranges}
        info(f'training: {split_seqs_nr["train"]}, '
             f'testing: {split_seqs_nr["test"]}, '
             f'validation: {split_seqs_nr["val"]}')
        return (BatchGenerator(*self.get_train_files(), **kwargs),
                BatchGenerator(*self.get_val_files(), **kwargs),
                BatchGenerator(*self.get_test_files(), **kwargs))


@dataclass
class BatchGenerator(Sequence):
    file_names: list
    labels: list
    classes: list
    batch_size: int
    rev_comp: bool = False
    rev_comp_mode: str = 'append'
    fixed_size_method: str = 'pad'
    enc_method: Callable[[str], list] = words2onehot
    enc_dimension: int = 64
    enc_k: int = 3
    enc_stride: int = 3
    max_seq_len: int = 10_000
    force_max_len: bool = True
    cache: bool = False         # cache batches
    cache_seq_limit: Optional[int] = None  # how many sequences to cache
    w2vfile: Optional[str] = None
    custom_encode_sequence: Optional[Callable[[str], list]] = None
    process_batch_function: Optional[Callable[[list], list]] = None

    def __post_init__(self):
        if (not self.force_max_len):
            raise Exception('not supported anymore')
            # info('determine max seq length')
            # self.det_max_seq_len()
        self.cached = {}

    def get_rev_comp(seq):
        return str(Seq(seq).reverse_complement())

    def get_seq(self, file_name):
        # unnecessary conditional, but small performance boost (maybe)
        if (self.rev_comp and self.rev_comp_mode == 'independent'):
            raw_seq = read_seq(re.sub(r'\$.*$', '', file_name))
        else:
            raw_seq = read_seq(file_name)
        if (self.custom_encode_sequence is not None):
            return self.custom_encode_sequence(raw_seq)
        if (self.rev_comp):
            try:
                if (self.rev_comp_mode == 'append'):
                    # TODO: maybe padding in between = 3*10 N's
                    raw_seq += BatchGenerator.get_rev_comp(raw_seq)
                elif (self.rev_comp_mode == 'random'):
                    raw_seq = choice((raw_seq,
                                      BatchGenerator.get_rev_comp(raw_seq)))
                elif (self.rev_comp_mode == 'independent'):
                    if (file_name.endswith('$rev_comp')):
                        debug(f'{file_name}\'s reverse complement is used')
                        raw_seq = BatchGenerator.get_rev_comp(raw_seq)
                else:
                    raise Exception(f'rev_comp_mode {self.rev_comp_mode} '
                                    'not supported')
            except ValueError as e:
                warning(f'rev_comp of sequence {file_name} could not be '
                        f'computed: {e}')
        method_kwargs = {}
        if (self.enc_method == words2index):
            method_kwargs['handle_nonalph'] = 'special'
        elif (self.enc_method == words2onehot):
            method_kwargs['handle_nonalph'] = 'split'
        elif (self.enc_method == words2vec):
            method_kwargs['w2vfile'] = self.w2vfile
        return np.array(encode_sequence(
            raw_seq, fixed_size_method=self.fixed_size_method,
            max_seq_len=self.max_seq_len,
            method=self.enc_method,
            k=self.enc_k,
            stride=self.enc_stride,
            **method_kwargs))

    def __len__(self):
        return np.ceil(len(self.file_names) /
                       float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        batch_filenames = self.file_names[idx * self.batch_size:
                                          (idx+1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:
                                   (idx+1) * self.batch_size]

        class_vectors = get_class_vectors(self.classes)
        if (self.cache and idx in self.cached):
            return self.cached[idx]
        batch_x = [self.get_seq(file_name) for file_name in batch_filenames]
        batch_y = np.array([class_vectors[label] for label in batch_labels])
        if (self.process_batch_function is not None):
            batch_x = self.process_batch_function(batch_x)
        result = (np.array(batch_x), batch_y)
        if (self.cache):
            if (self.cache_seq_limit is not None
                and len(self.cached) * self.batch_size
                <= self.cache_seq_limit):
                self.cached[idx] = result
        return result


def gen_files():
    # dump sequence file names
    split = DataSplit('../../sequences/dna_sequences/', 10,
                      ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'])
    split.store_seq_file_names('../../sequences/dna_sequences/files.json',
                               'json')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    gen_files()
