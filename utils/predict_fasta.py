import argparse
import keras
import keras_bert
from Bio.SeqIO import parse
from models.bert_utils import seq2tokens, get_token_dict, process_bert_tokens_batch
from models.model import PARAMS
import numpy as np
from os.path import splitext


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
    out_prefix = splitext(args.fasta)[0]
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(args.model, custom_objects=custom_objects)
    seq_len = args.seq_len
    token_dict = get_token_dict()
    records = list(parse(args.fasta, 'fasta'))
    x = process_bert_tokens_batch([seq2tokens(str(r.seq), token_dict, seq_len) for r in records])
    pred = model.predict(x, verbose=0)
    np.save(out_prefix + '_preds', pred)
    print('\t'.join([f'{c}: {i}' for i, c in enumerate(classes)]))
    for r, y in zip(records, pred):
        print(f'species: {r.id}\t prediction: {y}')
