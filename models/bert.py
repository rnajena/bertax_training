from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
from itertools import product, combinations
from random import shuffle
from collections import deque
from math import ceil
from process_inputs import kmer2index, ALPHABET, read_seq, seq2kmers
from generate_data import DataSplit
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks as keras_cbs
# from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(
#     intra_op_parallelism_threads=16,
#     inter_op_parallelism_threads=16)))

batch_size = 70
val_steps = 10

import sys

root_fa_dir = sys.argv[1]
from_cache = sys.argv[2]
min_split = 50
max_split = 250
head_num = 5
transformer_num = 12
embed_dim = 25
feed_forward_dim = 100
seq_len = max_split
pos_num = max_split
dropout_rate = 0.05


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
# normal 3mers
for word in [''.join(_) for _ in product(ALPHABET, repeat=3)]:
    token_dict[word] = len(token_dict)
# special 3mers: all special index
special_letters = 'NYRSWKMBDHV'
special_index = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word
# NOTE: special words are NOT in token_list!!!
for word in [''.join(_) for _ in product(ALPHABET + special_letters, repeat=3)]:
    if (all(letter in ALPHABET for letter in word)):
        continue
    else:
        token_dict[word] = special_index

# import json
# json.dump(token_dict, open('keras-bert_token_dict.json', 'w'))

# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=head_num,
    transformer_num=transformer_num,
    embed_dim=embed_dim,
    feed_forward_dim=feed_forward_dim,
    seq_len=seq_len,
    pos_num=pos_num,
    dropout_rate=dropout_rate,
)
compile_model(model)
model.summary()

pairs = deque()


def opt_split(n, min_, max_):
    min_x = n // max_
    max_x = n // min_
    if (max_x <= 2):
        return 2
    if (min_x % 2 == 1):
        min_x += 1
    # c is a factor that achieves convergence of n / opt_split to
    # (min_ + max_) / 2
    c = (min_ + max_)**2 / (2 * min_ * max_)
    x = (max_x + max(min_x, 2)) // c
    if (x % 2 == 0):
        return x
    else:
        if (x + 1 <= max_x):
            return x + 1
        elif (x - 1 >= min_x):
            return x - 1
        elif (min_x - 1 >= 2):
            return min_x - 1
        return 2


def seq_split_generator(seq, split_min, split_max):
    step = ceil(len(seq) / opt_split(len(seq), split_min, split_max))
    i = 0
    while (i < len(seq)):
        yield(seq[i:i + step])
        i += step


class FileNameGenerator:
    def __init__(self, files):
        # split = DataSplit(root_fa_dir, 100,
        #                   ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
        #                   from_cache, balance=True, train_test_split=0,
        #                   val_split=0.05)
        self.file_names = files
        self.i = 0

    def __len__(self):
        return len(self.file_names)

    def __iter__(self):
        return self

    def __next__(self):
        if (self.i < len(self.file_names)):
            file_name = self.file_names[self.i]
            self.i += 1
            return file_name
        else:
            raise StopIteration()


def add_seq(pairs, file_names):
    seq = seq2kmers(read_seq(next(file_names)), k=3, stride=3,
                    pad=True)
    seq_sentences = [sentence for sentence in
                     seq_split_generator(seq, min_split, max_split)]
    i = 1
    while i <= len(seq_sentences):
        pair = seq_sentences[i-1:i+1]
        if (len(pair) == 1):
            pair.append([])
        pairs.append(pair)
        i += 2
    if (len(pairs) == 0):
        # empty sequence or what?
        print(seq, seq_sentences)
        raise StopIteration()


class PairBatchGenerator(Sequence):
    def __init__(self, files):
        self.files = files
        self.pairs = deque()
        self.files_gen = FileNameGenerator(self.files)
        self.i = 0

    def batch_inputs(batch):
        return(gen_batch_inputs(
            batch,
            token_dict,
            token_list,
            seq_len=max_split,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        ))

    def __len__(self):
        # lower bound for now, exact calculation expensive
        return len(self.files) // batch_size

    def __getitem__(self, idx):
        if (self.i % len(self) == 0):
            self.files_gen = FileNameGenerator(self.files)
            self.pairs.clear()
        self.i += 1
        batch = []
        for i in range(batch_size):
            if (len(self.pairs) == 0):
                add_seq(self.pairs, self.files_gen)
            batch.append(self.pairs.popleft())
        return PairBatchGenerator.batch_inputs(batch)

split = DataSplit(root_fa_dir, 250_000,
                  ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
                  from_cache, balance=True, train_test_split=0,
                  val_split=0.05)
train_g = PairBatchGenerator(split.get_train_files()[0])
val_g = PairBatchGenerator(split.get_val_files()[0])

model.fit_generator(
    generator=train_g,
    epochs=100,
    validation_data=val_g,
    callbacks=[
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        # keras_cbs.ModelCheckpoint('bert.h5')
    ],
)

model.save('bert.h5')
exit(0)

# Use the trained model
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=head_num,
    transformer_num=transformer_num,
    embed_dim=embed_dim,
    feed_forward_dim=feed_forward_dim,
    seq_len=seq_len,
    pos_num=pos_num,
    dropout_rate=dropout_rate,
    training=False,      # The input layers and output layer will be returned if `training` is `False`
    trainable=False,     # Whether the model is trainable. The default value is the same with `training`
    output_layer_num=4,  # The number of layers whose outputs will be concatenated as a single output.
                         # Only available when `training` is `False`.
)
