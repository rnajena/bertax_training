import numpy as np
import json
import os.path
from random import shuffle, sample
from sklearn.model_selection import train_test_split
from preprocessing.process_inputs import seq2kmers, ALPHABET
from preprocessing.generate_data import load_fragments
from models.model import PARAMS
from keras_bert import get_model, compile_model
from keras_bert import gen_batch_inputs
import argparse
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from models.bert_utils import get_token_dict
from utils.tax_entry import TaxidLineage

class FragmentGenerator(Sequence):

    def __init__(self, fragments, species, seq_len, class_weights,
                 cache_lineage=True):
        self.fragments = fragments
        self.species = species
        self.seq_len = seq_len
        self.tlineage = TaxidLineage()
        self.weight_classes = class_weights
        if (cache_lineage):
            self.tlineage.populate(species, ['superkingdom'])

    def get_sample_weight(self, taxid):
        weight = 0
        ranks = self.tlineage.get_ranks(taxid, ranks=['superkingdom'])
        superkingdom = ranks['superkingdom'][1]
        # calc sample weight
        weight += self.weight_classes['superkingdom'].get(superkingdom,
                                                          self.weight_classes['superkingdom']['unknown'])
        # weight += self.weight_classes['kingdom'].get(kingdom,self.weight_classes['kingdom']['unknown'])
        # weight += self.weight_classes['family'].get(family,self.weight_classes['family']['unknown'])
        return weight

    def __len__(self):
        global batch_size
        return np.ceil(len(self.fragments) /
                       float(batch_size)).astype(np.int)

    def __getitem__(self, idx):
        global token_dict, token_list
        batch_fragments = self.fragments[idx * batch_size:
                                         (idx+1) * batch_size]
        batch_species = self.species[idx * batch_size:
                                     (idx+1) * batch_size]
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False, to_upper=True)
                      for seq in batch_fragments]
        sentences = [[seq[:len(seq)//2], seq[len(seq)//2:]]
                     for seq in batch_seqs]
        return (*gen_batch_inputs(sentences, token_dict, token_list,
                                  seq_len=self.seq_len),
                np.array(list(map(self.get_sample_weight, batch_species))))
        return gen_batch_inputs(sentences, token_dict, token_list,
                                seq_len=self.seq_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='pre-train BERT on pre-generated fragments')
    parser.add_argument('fragments_dir')
    parser.add_argument('--nr_seqs', help=' ', type=int, default=300_000)
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--batch_size', help=' ', type=int, default=70)
    parser.add_argument('--val_split', help=' ', type=float, default=0.02)
    parser.add_argument('--head_num', help=' ', type=int, default=12)
    parser.add_argument('--transformer_num', help=' ', type=int, default=12)
    parser.add_argument('--embed_dim', help=' ', type=int, default=768)
    parser.add_argument('--feed_forward_dim', help=' ', type=int, default=3072)
    parser.add_argument('--dropout_rate', help=' ', type=float, default=0.1)
    parser.add_argument('--epochs', help=' ', type=int, default=50)
    parser.add_argument('--no_balance', help=' ', action='store_true')
    parser.add_argument('--name', help=' ', type=str, default='bert_nc')
    args = parser.parse_args()
    batch_size = args.batch_size
    # building model
    token_dict = get_token_dict(ALPHABET, k=3)
    token_list = list(token_dict)
    model = get_model(
        token_num=len(token_dict),
        head_num=args.head_num,
        transformer_num=args.transformer_num,
        embed_dim=args.embed_dim,
        feed_forward_dim=args.feed_forward_dim,
        seq_len=args.seq_len,
        pos_num=args.seq_len,
        dropout_rate=args.dropout_rate)
    compile_model(model)
    model.summary()
    classes = PARAMS['data']['classes'][1]
    # loading training data
    fragments, y, species = load_fragments(
        args.fragments_dir, classes=classes,
        shuffle_=True, balance=(not args.no_balance),
        nr_seqs=args.nr_seqs)
    f_train, f_val, s_train, s_val, c_train, c_val = train_test_split(
        fragments, species, y, test_size=args.val_split)
    # f_train = [''.join(random_words(10)) for i in range(100)]
    # f_val = [''.join(random_words(10)) for i in range(10)]
    train_weights = {'superkingdom': {c: 1 / (len([yi for yi in c_train if yi == c]) / len(c_train))
                          for c in classes}}
    train_weights['superkingdom']['unknown'] = 1.0
    val_weights = {'superkingdom': {c: 1 / (len([yi for yi in c_val if yi == c]) / len(c_val))
                                    for c in classes}}
    val_weights['superkingdom']['unknown'] = 1.0
    train_gen = FragmentGenerator(f_train, s_train, args.seq_len, train_weights)
    val_gen = FragmentGenerator(f_val, s_val, args.seq_len, val_weights)
    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[ModelCheckpoint(args.name + '_ep{epoch:02d}.h5')])
    model.save(args.name + '_trained.h5')
