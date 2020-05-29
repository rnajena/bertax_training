import argparse
import keras
from Bio.SeqIO import parse
from models.bert_utils import seq2tokens, get_token_dict
from models.model import PARAMS


classes = PARAMS['data']['classes'][1]


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model')
    parser.add_argument('fasta')
    parser.add_argument('--conf_matrix', action='store_true')
    parser.add_argument('--seq_len', type=int, default=502)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model = keras.models.load_model(args.model)
    seq_len = args.seq_len
    token_dict = get_token_dict()
    records = list(parse(args.fasta, 'fasta'))
    x = [seq2tokens(str(r.seq), token_dict, seq_len) for r in records]
    pred = model.predict(x, verbose=1)
    print('\t'.join([f'{c}: {i}' for i, c in enumerate(classes)]))
    for r, y in zip(records, pred):
        print(f'species: {r.id}\t prediction: {y}')
