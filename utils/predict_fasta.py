import argparse
from Bio.SeqIO import parse
from models.bert_utils import seq2tokens, get_token_dict
from models.bert_utils import process_bert_tokens_batch, load_bert
from models.model import PARAMS
import numpy as np
from os.path import splitext
from logging import warning


classes = PARAMS['data']['classes'][1]


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model')
    parser.add_argument('fasta')
    parser.add_argument('--conf_matrix', action='store_true')
    parser.add_argument('--seq_len', type=int, default=502)
    parser.add_argument('--seq_len_like',
                        help='path of class dict of seq lens for generating '
                        'sampled sequence sizes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    out_prefix = splitext(args.fasta)[0]
    model = load_bert(args.model)
    max_seq_len = model.input_shape[0][1]
    if (args.seq_len > max_seq_len):
        warning(f'provided seq len ({args.seq_len}) exceeds possible maximum '
                f'seq len ({max_seq_len}). seq len will be adapted to maximum')
        args.seq_len = max_seq_len
    seq_len = args.seq_len
    token_dict = get_token_dict()
    records = list(parse(args.fasta, 'fasta'))
    x = process_bert_tokens_batch([seq2tokens(str(r.seq), token_dict, seq_len) for r in records])
    pred = model.predict(x, verbose=0)
    np.save(out_prefix + '_preds', pred)
    print('\t'.join([f'{c}: {i}' for i, c in enumerate(classes)]))
    for r, y in zip(records, pred):
        print(f'species: {r.id}\t prediction: {y}')
