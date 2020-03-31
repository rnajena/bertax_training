if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import json
import os.path
from random import shuffle
from sklearn.model_selection import train_test_split
from preprocessing.process_inputs import seq2kmers, ALPHABET
from keras_bert import get_base_dict, get_model, compile_model
from keras_bert import gen_batch_inputs
from argparse import ArgumentParser
from itertools import product
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from logging import warning


def load_fragments(fragments_dir, shuffle_=True):
    fragments = []
    for class_ in ('Archaea', 'Viruses', 'Eukaryota', 'Bacteria'):
        fragments.extend(json.load(open(os.path.join(
            fragments_dir, f'{class_}_fragments.json'))))
    if (shuffle_):
        shuffle(fragments)
    return fragments


class FragmentGenerator(Sequence):

    def __init__(self, fragments, seq_len):
        self.fragments = fragments
        self.seq_len = seq_len

    def __len__(self):
        global batch_size
        return np.ceil(len(self.fragments) /
                       float(batch_size)).astype(np.int)

    def __getitem__(self, idx):
        global token_dict, token_list
        batch_fragments = self.fragments[idx * batch_size:
                                         (idx+1) * batch_size]
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False).upper()
                      for seq in batch_fragments]
        sentences = [[seq[:len(seq)//2], seq[len(seq)//2:]]
                     for seq in batch_seqs]
        return gen_batch_inputs(sentences, token_dict, token_list,
                                seq_len=self.seq_len)


def get_token_dict(alph=ALPHABET, k=3):
    token_dict = get_base_dict()
    for word in [''.join(_) for _ in product(ALPHABET, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict


if __name__ == '__main__':
    parser = ArgumentParser(
        description='pre-train BERT on pre-generated fragments')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', type=int, default=502)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--val_split', type=float, default=0.02)
    parser.add_argument('--head_num', type=int, default=12)
    parser.add_argument('--transformer_num', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    # args = parser.parse_args(['/home/lo63tor/master/dna_class/output/genomic_fragments/',
    #                           '--seq_len', '502', '--batch_size', '70', '--head_num', '2',
    #                           '--transformer_num', '2', '--embed_dim', '10', '--epochs', '3'])
    batch_size = args.batch_size
    # building model
    token_dict = get_token_dict(ALPHABET, k=3)
    token_list = list(token_dict)
    model = get_model(
        token_num=len(token_dict),
        head_num=args.head_num,
        transformer_num=args.transformer_num,
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        pos_num=args.seq_len)
    compile_model(model)
    model.summary()
    # loading training data
    fragments = load_fragments(args.fragments_dir)
    f_train, f_val = train_test_split(fragments, test_size=args.val_split)
    # f_train = [''.join(random_words(10)) for i in range(100)]
    # f_val = [''.join(random_words(10)) for i in range(10)]
    model.fit_generator(
        generator=FragmentGenerator(f_train, args.seq_len),
        epochs=args.epochs,
        validation_data=FragmentGenerator(f_val, args.seq_len),
        callbacks=[ModelCheckpoint('bert_nc_ep{epoch:02d}.h5')])
    model.save('bert_nc_trained.h5')
