import numpy as np
import json
import os.path
from random import shuffle
from sklearn.model_selection import train_test_split
from preprocessing.process_inputs import seq2kmers, ALPHABET
from keras_bert import get_model, compile_model
from keras_bert import gen_batch_inputs
import argparse
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from models.bert_utils import get_token_dict


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
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False, to_upper=True)
                      for seq in batch_fragments]
        sentences = [[seq[:len(seq)//2], seq[len(seq)//2:]]
                     for seq in batch_seqs]
        return gen_batch_inputs(sentences, token_dict, token_list,
                                seq_len=self.seq_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='pre-train BERT on pre-generated fragments')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--batch_size', help=' ', type=int, default=70)
    parser.add_argument('--val_split', help=' ', type=float, default=0.02)
    parser.add_argument('--head_num', help=' ', type=int, default=12)
    parser.add_argument('--transformer_num', help=' ', type=int, default=12)
    parser.add_argument('--embed_dim', help=' ', type=int, default=768)
    parser.add_argument('--feed_forward_dim', help=' ', type=int, default=3072)
    parser.add_argument('--dropout_rate', help=' ', type=float, default=0.1)
    parser.add_argument('--epochs', help=' ', type=int, default=50)
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
    # loading training data
    fragments = load_fragments(args.fragments_dir)
    f_train, f_val = train_test_split(fragments, test_size=args.val_split)
    # f_train = [''.join(random_words(10)) for i in range(100)]
    # f_val = [''.join(random_words(10)) for i in range(10)]
    model.fit(
        generator=FragmentGenerator(f_train, args.seq_len),
        epochs=args.epochs,
        validation_data=FragmentGenerator(f_val, args.seq_len),
        callbacks=[ModelCheckpoint(args.name + '_ep{epoch:02d}.h5')])
    model.save(args.name + '_trained.h5')
