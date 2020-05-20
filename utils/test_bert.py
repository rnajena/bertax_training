import argparse
from preprocessing.process_inputs import ALPHABET
from preprocessing.generate_data import DataSplit, BatchGenerator
from models.model import PARAMS
from models.bert_utils import get_token_dict
from models.bert_utils import seq2tokens, process_bert_tokens_batch
from models.bert_utils import load_bert
from models.bert_nc_finetune import load_fragments, FragmentGenerator
import numpy as np
from logging import warning, getLogger, DEBUG
from os.path import splitext, basename
from time import time
import pickle

SOURCES = ['genes', 'fragments', 'fasta']


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')
    parser.add_argument('model_path')
    parser.add_argument('--source', default='genes',
                        choices=SOURCES, help=' ')
    parser.add_argument('--store_predictions', action='store_true', help=' ')
    parser.add_argument('--fasta', default=None,
                        help='fasta to load if source=fasta')
    parser.add_argument('--conf_matrix', action='store_true', help=' ')
    parser.add_argument('--seq_len', type=int, default=502,
                        help='fixed length for all sequences')
    parser.add_argument('--seq_len_like', default=None,
                        help='path of pickled class dict of seq lens for '
                        'generating sampled sequence sizes')
    parser.add_argument('--no_seq_len_window', action='store_true',
                        help='if specified, the beginning of the sequence'
                        'will be used instead of a random window, if the '
                        'sequence is too long')
    parser.add_argument('--nr_seqs', default=250_000, type=int,
                        help='nr of sequences to use per class')
    parser.add_argument('--batch_size', default=32, type=int,
                        help=' ')
    parser.add_argument('--root_fa_dir', help=' ',
                        default=PARAMS['data']['root_fa_dir'][1])
    parser.add_argument('--from_cache', help=' ',
                        default=PARAMS['data']['file_names_cache'][1])
    parser.add_argument('--fragments_dir', help=' ',
                        default='/home/lo63tor/master/sequences/dna_sequences/fragments/genomic_fragments_80')
    parser.add_argument('--no_balance', help=' ', action='store_true')
    parser.add_argument('--repeated_undersampling', help=' ', action='store_true')
    parser.add_argument('--classes', help=' ', nargs='+',
                        default=PARAMS['data']['classes'][1])
    parser.add_argument('--alphabet', help=' ', default=ALPHABET)
    parser.add_argument('--k', help=' ', default=3, type=int)
    parser.add_argument('--stride', help=' ', default=3, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    getLogger().setLevel(DEBUG)
    args = parse_arguments()
    token_dict = get_token_dict(args.alphabet, args.k)
    model = load_bert(args.model_path, compile_=True)
    max_seq_len = model.input_shape[0][1]
    if (args.seq_len > max_seq_len):
        warning(f'provided seq len ({args.seq_len}) exceeds possible maximum '
                f'seq len ({max_seq_len}). seq len will be adapted to maximum')
        args.seq_len = max_seq_len
    seq_len = args.seq_len
    if (args.seq_len_like is not None):
        seq_len_dict = pickle.load(open(args.seq_len_like, 'rb'))
        min_nr_seqs = min(map(len, seq_len_dict.values()))
        seq_len_like = []
        for k in seq_len_dict:
            seq_len_like.extend(np.random.choice(seq_len_dict[k], min_nr_seqs)
                                // args.k)
    else:
        seq_len_like = None

    encode_kwargs = {'k': args.k, 'stride': args.stride,
                     'seq_len_like': seq_len_like}

    def custom_encode_sequence(seq):
        return seq2tokens(seq, token_dict, seq_length=seq_len,
                          max_length=max_seq_len,
                          window=True, **encode_kwargs)

    if (args.source == 'genes'):
        split = DataSplit(root_fa_dir=args.root_fa_dir, nr_seqs=args.nr_seqs,
                          classes=args.classes, from_cache=args.from_cache,
                          train_test_split=0, val_split=0,
                          balance=not args.no_balance,
                          repeated_undersampling=args.repeated_undersampling)
        x, y = split.get_train_files()
        generator = BatchGenerator(
            x, y, args.classes, args.batch_size,
            custom_encode_sequence=custom_encode_sequence,
            process_batch_function=process_bert_tokens_batch, enc_k=args.k,
            enc_stride=args.stride, save_batches=True)
    elif (args.source == 'fragments' or args.source == 'fasta'):
        if (args.source == 'fragments'):
            x, y = load_fragments(args.fragments_dir, balance=not args.no_balance,
                                  nr_seqs=args.nr_seqs)
        else:
            from Bio.SeqIO import parse
            records = list(parse(args.fasta, 'fasta'))
            x = [str(r.seq) for r in records]
            if (records[0].id in args.classes):
                # fasta IDs ^= classes
                y = [r.id for r in records]
            else:
                y = []
        generator = FragmentGenerator(x, y, seq_len,
                                      max_seq_len=max_seq_len,
                                      batch_size=args.batch_size,
                                      classes=args.classes,
                                      window=(not args.no_seq_len_window),
                                      **encode_kwargs)
    else:
        raise Exception(f'only sources {SOURCES} are accepted')
    if (not args.store_predictions and len(y) != 0):
        results = model.evaluate(generator)
    else:
        preds = model.predict(generator, verbose=1)
        preds_discrete = np.argmax(preds, axis=1)
        if (args.source == 'genes'):
            # when BatchGenerator is used, batches are randomized,
            # thus x and y have to be reloaded
            x = []
            y = []
            for nr, files, labels, result in generator.stored:
                x.extend(files)
                y.extend(labels)
        y_indices = list(map(args.classes.index, y))
        results = [np.nan,
                   np.sum(preds_discrete == y_indices) / len(y)
                   if len(y) != 0 else np.nan]
        filepath = (splitext(basename(args.model_path))[0] + '_'
                    + str(int(time())))
        pickle.dump({'classes': args.classes, 'x': x, 'y': y_indices,
                     'preds': preds}, open(filepath + '.pkl', 'wb'))
    print('results:', results)
