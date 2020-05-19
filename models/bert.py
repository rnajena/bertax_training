if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import os
os.environ['TF_KERAS']="1"
from keras_bert import get_base_dict, get_model, compile_model
from keras_bert import gen_batch_inputs
from itertools import product
from math import ceil
from preprocessing.process_inputs import ALPHABET, read_seq, seq2kmers
from preprocessing.generate_data import DataSplit
from tqdm import tqdm
import sys
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from random import sample
from math import log10

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
mirrored_strategy = tf.distribute.MirroredStrategy()

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
seq_len = 512                   # ~= double max_split
head_num = 12                   # =:A
transformer_num = 12            # =:L
embed_dim = 768                 # =:H (NOTE: has to be dividable by A)
feed_forward_dim = 3072         # default
pos_num = seq_len
dropout_rate = 0.1              # default


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


def run_epoch(filenames, model_function, progress_bar=False):
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
        filenames = tqdm(filenames)
    for filename in filenames:
        seq = seq2kmers(read_seq(filename), k=3, stride=3,
                        pad=True)
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

def batch_generator(filenames):
    # while True:
    order = np.arange(len(filenames))
    order = np.random.permutation(order)
    pairs = []
    i = 0
    pbar = tqdm(total=len(filenames))
    while (i < len(filenames)):
        # seq = seq2kmers(read_seq_ssd_version(root_fa_dir, filenames[order[i]]), k=1,stride=1)
        seq = seq2kmers(read_seq(filenames[order[i]]), k=3,stride=3)
        seq_sentences = [sentence for sentence in
                         seq_split_generator(seq, min_split, max_split)]
        pairs.extend(zip(*[iter(seq_sentences)] * 2))
        while (len(pairs) >= batch_size):
            yield gen_batch_inputs(
                pairs[:batch_size],
                token_dict,
                token_list,
                seq_len=seq_len)
            pairs = pairs[batch_size:]

        i += 1
        pbar.update(1)
    pbar.close()
    print("end of epoch")

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
    token_dict = get_token_dict()
    token_list = list(token_dict)

    # Build & train the model ready for multi-gpu
    with mirrored_strategy.scope():
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
                      from_cache, balance=balance, train_test_split=0,
                      val_split=val_split, repeated_undersampling=True)
    print('split done')
    files_train = split.get_train_files()[0]
    files_val = split.get_val_files()[0]
    train_g = batch_generator(filenames=files_train)
    val_g = batch_generator(filenames=files_val)
    test_g = val_g

    filepath1 = "model.best.acc.hdf5"
    filepath2 = "model.best.loss.hdf5"
    # checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True,
    #                               save_weights_only=False, mode='max')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='min')
    checkpoint3 = EarlyStopping('val_loss', min_delta=0, patience=0, restore_best_weights=True)
    callbacks_list = [checkpoint2, checkpoint3]

    try:
        for epoch in range(epochs):
            model.fit(train_g, validation_data=val_g, verbose=2, steps_per_epoch=None,validation_steps=None,
                       epochs=epoch+1,initial_epoch=epoch, callbacks=callbacks_list)
            train_g = batch_generator(filenames=files_train)
            val_g = batch_generator(filenames=files_val)

    except (KeyboardInterrupt):
        print("training interrupted, current status will be saved and tested, press ctrl+c to cancel this")
        file_suffix = '_aborted.hdf5'
        model.save("bert_small_trained" + file_suffix)
        print('testing...')
        result = model.evaluate(test_g)
        print("test results:",*zip(model.metrics_names, result))
        exit()

    print('testing...')
    result = model.evaluate(test_g,steps=(len(files_val)))
    print("test results:", *zip(model.metrics_names, result))
    model.save(f'bert_small_trained.hdf5')

def random_words(n):
    from random import choices
    return choices([''.join(_) for _ in product(ALPHABET, repeat=3)], k=n)


def memory_batch(seq_len, batch_size):
    A = seq_len * batch_size * 8
    B = batch_size * 8
    return 3*A + A + B

# memory_batch(512, 250)
