import argparse
import model
from generate_data import DataSplit
from process_inputs import words2index, words2onehot
from tensorflow.keras.callbacks import EarlyStopping
import logging
from itertools import product


def str2bool(v):
    """allows reading in bools in argparse"""
    if type(v) == bool:
        return v
    try:
        return {'False': False, 'True': True}[v]
    except KeyError:
        raise argparse.ArgumentTypeError('Boolean value expected {True|False}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description='runs models')
    parser.add_argument('type', choices=['cnn', 'lstm', 'tcn'],
                        type=str,
                        help='(implemented) type of model to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='more detailed output [Default: False]')
    # add optional arguments
    for group, group_dict in model.PARAMS.items():
        g = parser.add_argument_group(group)
        for name, options in group_dict.items():
            a_nargs = '+'
            if (type(options[0]) == tuple):
                a_type = options[0][1]
            else:
                a_type = options[0]
            if (a_type == bool):
                a_type = str2bool
            a_default = options[1]
            a_model_type = None
            if (len(options) > 2 and options[2] is not None):
                a_model_type = options[2]
            a_help = (f'only applicable to {a_model_type}'
                      if 'a_model_type' in locals()
                      and a_model_type is not None else '')
            if (len(options) > 3):
                a_help = options[3]
            a_choices = None
            if (len(options) > 4):
                a_choices = options[4]
            a_help += (' [Default: '
                       f'{(a_default if a_default is not None else None)}]')
            g.add_argument(f'--{name}', type=a_type, default=a_default,
                           help=a_help, choices=a_choices, nargs=a_nargs)
    args = parser.parse_args()
    from pprint import pprint
    pprint(args)
    for key, val in vars(args).items():
        # TODO: dont transform params that are intended to be lists
        # for group in model.PARAMS:
        #     if (key in model.PARAMS[group] and type(
        #             model.PARAMS[group][0]) == tuple):
        #         continue
        if type(val) == list and len(val) == 1:
            vars(args)[key] = val[0]
    # store raw arguments (no functions) for json serialization
    args_repr = dict(vars(args))
    args.enc_method = {'words2index': words2index, 'words2onehot':
                       words2onehot}[args.enc_method]
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] %(message)s')
    # validate arguments
    if (args.type == 'tcn'):
        import tensorflow as tf
        if (not tf.__version__.startswith('1.')):
            raise Exception('You seem to be using tensorflow > version 1. '
                            'TCN models only work with tensorflow<2')
    if (args.enc_stride > args.enc_k):
        logging.warning(f'The k-mer stride (f{args.enc_stride}) is larger '
                        f'than k ({args.enc_k}). Information will be lost!')
    if (args.verbose):
        logging.getLogger().setLevel(logging.DEBUG)
        print(args_repr)
    # general settings
    logging.info('splitting and balancing dataset')
    split = DataSplit(args.root_fa_dir, args.nr_seqs,
                      args.classes,
                      from_cache=args.file_names_cache,
                      train_test_split=args.test_split)
    train_g, val_g, test_g = split.to_generators(
        batch_size=args.batch_size, rev_comp=args.rev_comp,
        rev_comp_mode=args.rev_comp_mode,
        enc_method=args.enc_method,
        enc_dimension=args.enc_dimension,
        enc_k=args.enc_k,
        enc_stride=args.enc_stride,
        max_seq_len=args.max_seq_len,
        cache=args.cache_batches,
        cache_seq_limit=args.cache_seq_limit)
    callbacks = []
    if (args.early_stopping):
        logging.info('enabling early splitting')
        early_stopping = EarlyStopping(
            min_delta=args.early_stopping_md, patience=args.early_stopping_p,
            restore_best_weights=args.early_stopping_restore_weights)
        callbacks.append(early_stopping)
    m = model.DCModel(
            classes=args.classes, max_seq_len=args.max_seq_len,
            enc_dimension=args.enc_dimension, name=args.model_name)
    # model-specific settings
    model_settings = {}
    for key, spec in model.PARAMS['nns'].items():
        if (not (len(spec) < 3 or args.type in spec[2])):
            continue
        val = vars(args)[key]
        if type(val) != list:
            val = [val]
        model_settings[key] = val
    keys = list(model_settings.keys())
    combinations = [dict(zip(keys, prod)) for prod in
                    product(*(model_settings[key] for key in keys))]
    logging.info(f'creating model architecture: f{args.type}')
    gen_model_fn = {'cnn': m.generate_cnn_model,
                    'lstm': m.generate_lstm_model,
                    'tcn': m.generate_tcn_model}[args.type]
    if (len(combinations) > 1):
        logging.info('some parameters have multiple values, '
                     'all combinations will be run')
    for i, settings in enumerate(combinations):
        # name = {given_name}_{hash_of_settings}
        import pickle
        from zlib import adler32
        m.name = f'{args.model_name}_{adler32(pickle.dumps(settings))}'
        if (args.type == 'tcn'):
            settings['nb_filters'] = settings['nr_filters']
            del settings['nr_filters']
            settings['dilations'] = [2 ** i for i in range(
                settings['last_dilation_2exp'])]
            del settings['last_dilation_2exp']
        print(f'model: {m.name}, settings: {settings}')
        gen_model_fn(**settings)
        logging.info('training')
        m.train(train_g, val_g, epochs=args.epochs, callbacks=callbacks)
        logging.info('evaluating')
        m.eval(test_g, args.class_report)
        m.write_to_file(m.name + '.json', args_repr, settings)
