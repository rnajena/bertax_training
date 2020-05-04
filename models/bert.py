if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from keras_bert import get_base_dict, get_model, compile_model
from keras_bert import gen_batch_inputs
from itertools import product
from math import ceil
from preprocessing.process_inputs import ALPHABET, read_seq_ssd_version, seq2kmers
from preprocessing.generate_data import DataSplit
from tqdm import tqdm
import sys
import numpy as np
from random import sample
from math import log10

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

root_fa_dir = sys.argv[1]
from_cache = sys.argv[2]
progress_bar = True

# controlling training sizes
balance = False
epochs = 25            # default
batch_size = 35              # default is 256, but probably too big for VRAM
# batch_size = 1                 # test
val_split = 0.005

# sentence splits
# chosen to correspond to average protein domain lengths
min_split = 50
max_split = 250

# bert parameters
# BERT_BASE (L=12, H=768, A=12)
seq_len = 500                   # ~= double max_split
head_num = 5                   # =:A 12
transformer_num = 12            # =:L 12
embed_dim = 25                 # =:H (NOTE: has to be dividable by A) 768
feed_forward_dim = 100         # default 3072
pos_num = seq_len
dropout_rate = 0.05              # default


def get_token_dict(alph=ALPHABET, k=3):
    token_dict = get_base_dict()
    for word in [''.join(_) for _ in product(ALPHABET, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict


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


def run_epoch(filenames, model_function, progress_bar=False,**kwargs):
    """trains on all filenames with an unknown amount of sentences(->steps)"""
    def train_batch(pairs):
        batch = gen_batch_inputs(
                pairs,
                token_dict,
                token_list,
                seq_len=seq_len)
        metrics = model_function(*batch, reset_metrics=False)
        return metrics
    metrics = None
    pairs = []
    if progress_bar:
        filenames = tqdm(filenames, smoothing=0.05)
    for filename in filenames:
        seq = seq2kmers(read_seq_ssd_version(root_fa_dir,filename),**kwargs)
        seq_sentences = [sentence for sentence in
                         seq_split_generator(seq, min_split, max_split)]
        pairs.extend(zip(*[iter(seq_sentences)]*2))
        if (len(pairs) >= batch_size):
            metrics = train_batch(pairs[:batch_size])
            pairs = pairs[batch_size:]
    if (len(pairs) > 0):
        for chunk in [pairs[i:i + batch_size]
                      for i in range(0, len(pairs), batch_size)]:
            metrics = train_batch(chunk)
    return metrics


if __name__ == '__main__':
    # import keras
    # keras.backend.set_floatx('float16')
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    # import tensorflow
    # tensorflow.keras.backend.set_floatx('float16')
    k = 1
    stride = 1
    kwargs = {"k":k, "stride":stride}

    token_dict = get_token_dict(k=k)
    token_list = list(token_dict)

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

    # NOTE: because of internal implementation: val_data := test_data
    split = DataSplit(root_fa_dir,
                      # 10,      # test
                      12_000_000,
                      ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
                      from_cache, balance=balance, train_test_split=val_split,
                      val_split=0)
    print('split done')
    files_train = split.get_train_files()[0]
    files_val = split.get_test_files()[0]
    for i in range(epochs):
        print(f'=== Epoch {i+1:2}/{epochs} ===')
        print('training')
        metrics = run_epoch(files_train, model.train_on_batch, progress_bar,**kwargs)
        print('training metrics',*zip(model.metrics_names, metrics))
        filename = f'bert_v1_epoch{i+1}.h5'
        print(f'saved to {filename}')
        model.save(filename)
        print('validating')
        metrics = run_epoch(files_val, model.test_on_batch,**kwargs)
        print('validation metrics', metrics)
    model.save(f'bert_v1_trained.h5')


def random_words(n):
    from random import choices
    return choices([''.join(_) for _ in product(ALPHABET, repeat=3)], k=n)


def memory_batch(seq_len, batch_size):
    A = seq_len * batch_size * 8
    B = batch_size * 8
    return 3*A + A + B

# memory_batch(512, 250)
