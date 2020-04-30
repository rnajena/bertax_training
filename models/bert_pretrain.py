from keras_bert import get_base_dict, get_model, compile_model
from keras_bert import gen_batch_inputs
from itertools import product
from math import ceil
from preprocessing.process_inputs import ALPHABET, read_seq, seq2kmers
from preprocessing.generate_data import DataSplit
from models.bert_utils import get_token_dict
from models.model import PARAMS
from tqdm import tqdm
import argparse


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
        batch = gen_batch_inputs(pairs,
                                 token_dict,
                                 token_list,
                                 seq_len=args.seq_len)
        metrics = model_function(*batch, reset_metrics=False)
        return metrics
    metrics = None
    pairs = []
    if progress_bar:
        filenames = tqdm(filenames)
    for filename in filenames:
        seq = seq2kmers(read_seq(filename), k=args.k, stride=args.stride,
                        pad=True)
        seq_sentences = [sentence for sentence in
                         seq_split_generator(seq, args.min_split,
                                             args.max_split)]
        pairs.extend(zip(*[iter(seq_sentences)] * 2))
        if (len(pairs) >= args.batch_size):
            metrics = train_batch(pairs[:args.batch_size])
            pairs = pairs[args.batch_size:]
    if (len(pairs) > 0):
        for chunk in [pairs[i:i + args.batch_size]
                      for i in range(0, len(pairs), args.batch_size)]:
            metrics = train_batch(chunk)
    return metrics

def log(*messages):
    global log_file
    print(*messages)
    with open(log_file, 'a') as f:
        f.write(f'[{time.strftime("%x %X")}]\t" ".join(messages)\n')


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')
    parser.add_argument('name', help='prefix for saved models')
    parser.add_argument('--root_fa_dir', help=' ',
                        default=PARAMS['data']['root_fa_dir'][1])
    parser.add_argument('--from_cache', help=' ',
                        default=PARAMS['data']['file_names_cache'][1])
    parser.add_argument('--no_progress_bar', help=' ', action='store_true')
    parser.add_argument('--no_balance', help=' ', action='store_true')
    parser.add_argument('--epochs', help=' ', type=int, default=15)            # default
    parser.add_argument('--batch_size', type=int, default=256,
                        help='decrease this for lower memory consumption')
    parser.add_argument('--val_split', help=' ', type=float, default=0.005)

    # sentence splits
    # chosen to correspond to average protein domain lengths
    parser.add_argument('--min_split', help=' ', type=int, default=50)
    parser.add_argument('--max_split', help=' ', type=int, default=250)

    # bert parameters
    # BERT_BASE (L=12, H=768, A=12)
    parser.add_argument('--seq_len', default=512, type=int,
                        help='should be at least `max_split`*2 + 2')
    parser.add_argument('--head_num', default=12, type=int,
                        help='=:A; BERT_BASE: 12, BERT_A: 5')
    parser.add_argument('--transformer_num', default=12, type=int,
                        help='=:L; BERT_BASE: 12, BERT_A: 12')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='=:H; BERT_BASE: 768, BERT_A: 25; '
                        'has to be dividable by A')
    parser.add_argument('--feed_forward_dim', default=3072, type=int,
                        help='BERT_BASE: 3072, BERT_A: 100')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='BERT_BASE: 0.1, BERT_A: 0.05')
    parser.add_argument('--nr_seqs', default=250_000, type=int,
                        help='nr of sequences to use per class')
    parser.add_argument('--classes', help=' ', default=PARAMS['data']['classes'][1])
    parser.add_argument('--alphabet', help=' ', default=ALPHABET)
    parser.add_argument('--k', help=' ', default=3 type=int)
    parser.add_argument('--stride', help=' ', type=int, default=3)
    args = parser.parse_args()
    args.pos_num = args.seq_len
    return args


if __name__ == '__main__':
    args = parse_arguments()
    token_dict = get_token_dict(alph=args.alphabet, k=args.k)
    token_list = list(token_dict)

    # Build & train the model
    model = get_model(
        token_num=len(token_dict),
        head_num=args.head_num,
        transformer_num=args.transformer_num,
        embed_dim=args.embed_dim,
        feed_forward_dim=args.feed_forward_dim,
        seq_len=args.seq_len,
        pos_num=args.pos_num,
        dropout_rate=args.dropout_rate)
    compile_model(model)
    model.summary()
    log_file = args.name + '_' + str(int(time.time())) + '.log'
    log('splitting...')
    # NOTE: because of internal implementation: val_data := test_data
    split = DataSplit(args.root_fa_dir,
                      args.nr_seqs,
                      args.classes,
                      args.from_cache, balance=not args.no_balance,
                      train_test_split=args.val_split,
                      val_split=0)
    log('split done')
    files_train = split.get_train_files()[0]
    files_val = split.get_test_files()[0]
    for i in range(args.epochs):
        log(f'=== Epoch {i+1:2}/{args.epochs} ===')
        log('training')
        metrics = run_epoch(files_train, model.train_on_batch,
                            not args.no_progress_bar)
        log('training metrics', metrics)
        filename = f'{args.name}_epoch{i+1}.h5'
        log(f'saved to {filename}')
        model.save(filename)
        log('validating')
        metrics = run_epoch(files_val, model.test_on_batch)
        log('validation metrics', metrics)
    model.save(f'{args.name}_trained.h5')
