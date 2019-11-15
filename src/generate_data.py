from os import scandir
import os.path
from random import sample, shuffle, choice
from process_inputs import words2onehot, words2index, seq2kmers, seq2nucleotides
from process_inputs import encode_sequence, read_seq, get_class_vectors
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.utils import Sequence
import logging
from logging import info, warning
import json
import pickle
from tqdm import tqdm
from typing import Optional, Callable
from Bio.Seq import Seq


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

    def __post_init__(self):
        self.get_fa_files(self.from_cache, self.from_cache_format)
        self.process_fa_files()
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
                    self.file_names, self.labels = json.load(f)
                    self.file_names = [os.path.join(self.root_fa_dir, f)
                                       for f in self.file_names]
            else:
                with open(from_cache, 'rb') as f:
                    self.file_names, self.labels = pickle.load(f)
        else:
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

    def process_fa_files(self):
        """shuffles and balances files"""
        # balance so that every class has {lowest_seq_nr} sequences
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
        for c in self.classes:
            for i in sample(class_indices[c], self.nr_seqs):
                file_names_b.append(self.file_names[i])
                labels_b.append(self.labels[i])
        # shuffle lists
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
                      enc_method=words2onehot, enc_dimension=64,
                      enc_k=3, enc_stride=3,
                      max_seq_len=10_000, force_max_len=True,
                      cache=False, cache_seq_limit=None) -> tuple:
        kwargs = {'classes': self.classes, 'batch_size': batch_size,
                  'rev_comp': rev_comp, 'rev_comp_mode': rev_comp_mode,
                  'enc_method': enc_method, 'enc_dimension': enc_dimension,
                  'enc_k': enc_k, 'enc_stride': enc_stride,
                  'max_seq_len': max_seq_len, 'force_max_len': force_max_len,
                  'cache': cache, 'cache_seq_limit': cache_seq_limit}
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
    enc_method: Callable[[str], list] = words2onehot
    enc_dimension: int = 64
    enc_k: int = 3
    enc_stride: int = 3
    max_seq_len: int = 10_000
    force_max_len: bool = True
    cache: bool = False         # cache batches
    cache_seq_limit: Optional[int] = None  # how many sequences to cache

    def __post_init__(self):
        if (not self.force_max_len):
            raise Exception('not supported anymore')
            # info('determine max seq length')
            # self.det_max_seq_len()
        self.cached = {}

    def get_seq(self, file_name):
        raw_seq = read_seq(file_name)
        if (self.rev_comp):
            seq_rc = str(Seq(raw_seq).reverse_complement())
            if (self.rev_comp_mode == 'append'):
                # TODO: maybe padding in between = 3*10 N's
                raw_seq += seq_rc
            elif (self.rev_comp_mode == 'random'):
                raw_seq = choice((raw_seq, seq_rc))
            else:
                raise Exception(f'rev_comp_mode {self.rev_comp_mode} '
                                'not supported')
        return encode_sequence(
            raw_seq, self.max_seq_len, pad=True,
            method=self.enc_method,
            k=self.enc_k,
            stride=self.enc_stride,
            handle_nonalph='special'
            if self.enc_method == words2index else 'split')

    def __len__(self):
        return np.ceil(len(self.file_names) /
                       float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.file_names[idx * self.batch_size:
                                  (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:
                              (idx+1) * self.batch_size]

        class_vectors = get_class_vectors(self.classes)
        if (self.cache and idx in self.cached):
            return self.cached[idx]
        result = (
            np.array([self.get_seq(file_name) for file_name in batch_x]),
            np.array([class_vectors[label] for label in batch_y]))
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
