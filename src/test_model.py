import argparse
import model
from generate_data import DataSplit
from process_inputs import words2index, words2onehot, words2vec
import logging
from itertools import product
import tensorflow as tf
if (tf.__version__.startswith('1.')):
    import keras.callbacks as keras_cbs
else:
    import tensorflow.keras.callbacks as keras_cbs


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
    parser.add_argument('type', choices=['cnn', 'cnndeep_predef', 'lstm',
                                         'tcn', 'ff'],
                        type=str,
                        help='(implemented) type of model to run')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no_tensorboard', action='store_false')
    # add optional arguments
    for group, group_dict in model.PARAMS.items():
        g = parser.add_argument_group(group)
        for name, options in group_dict.items():
            kw = {}
            kw['nargs'] = '+'
            a_raw_type = options[0]
            kw['default'] = options[1]
            if (len(options) > 2 and options[2] is not None):
                model_type = options[2]
            if (len(options) > 3):
                kw['help'] = options[3]
            else:
                kw['help'] = ''
            if (len(options) > 4):
                kw['choices'] = options[4]
            if (type(a_raw_type) == tuple):
                # e.g., 'classes': (((list, str), ['Viruses', 'Archaea']))
                kw['type'] = a_raw_type[1]
            else:
                # e.g., 'emb_layer_dim': (int, 1)
                kw['type'] = a_raw_type
            if ('help' == '' and 'model_type' in locals()
                and model_type is not None):
                kw['help'] = (f'only applicable to {model_type}')
            kw['help'] += (
                ' [Default: '
                f'{(kw["default"] if kw["default"] is not None else None)}]')
            if (kw['type'] == bool):
                if (kw['default'] is False):
                    del kw['type']
                    del kw['default']
                    del kw['nargs']
                    kw['action'] = 'store_true'
                else:
                    kw['type'] = str2bool
            g.add_argument(f'--{name}', **kw)
    args = parser.parse_args()
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
                       words2onehot, 'words2vec': words2vec}[args.enc_method]
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
    if (args.enc_method == words2vec):
        if (args.w2vfile is None):
            raise Exception(
                'When using the `words2vec` encoding method a words2vec'
                '(gensim) model filename has to be provided')
        words2vec.w2v = None
    if (args.verbose):
        logging.getLogger().setLevel(logging.DEBUG)
        from pprint import pprint
        pprint(args)
    # general settings
    logging.info('splitting and balancing dataset')
    split = DataSplit(args.root_fa_dir, args.nr_seqs,
                      args.classes,
                      from_cache=args.file_names_cache,
                      train_test_split=args.test_split)
    train_g, val_g, test_g = split.to_generators(
        batch_size=args.batch_size, rev_comp=args.rev_comp,
        rev_comp_mode=args.rev_comp_mode,
        fixed_size_method=args.fixed_size_method,
        enc_method=args.enc_method,
        enc_dimension=args.enc_dimension,
        enc_k=args.enc_k,
        enc_stride=args.enc_stride,
        max_seq_len=args.max_seq_len,
        cache=args.cache_batches,
        cache_seq_limit=args.cache_seq_limit,
        w2vfile=args.w2vfile)
    callbacks = []
    if (args.early_stopping):
        logging.info('enabling early stopping')
        early_stopping = keras_cbs.EarlyStopping(
            min_delta=args.early_stopping_md, patience=args.early_stopping_p,
            restore_best_weights=args.early_stopping_restore_weights)
        callbacks.append(early_stopping)
    m = model.DCModel(
        classes=args.classes, max_seq_len=args.max_seq_len,
        enc_dimension=args.enc_dimension, name=args.model_name,
        summary=args.summary, plot=args.plot, save=args.save)
    # model-specific settings
    model_settings = {}
    for key, spec in model.PARAMS['nns'].items():
        if (not (len(spec) < 3 or any(
                spec_type in args.type for spec_type in spec[2].split(',')))):
            continue
        val = vars(args)[key]
        if type(val) != list:
            val = [val]
        model_settings[key] = val
    keys = list(model_settings.keys())
    combinations = [dict(zip(keys, prod)) for prod in
                    product(*(model_settings[key] for key in keys))]
    # print(combinations)
    logging.info(f'creating model architecture: f{args.type}')
    gen_model_fn = {'cnn': m.generate_cnn_model,
                    'cnndeep_predef': m.generate_cnndeep_predef_model,
                    'lstm': m.generate_lstm_model,
                    'tcn': m.generate_tcn_model,
                    'ff': m.generate_ff_model}[args.type]
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
        print(f'configuration {i+1}/{len(combinations)}\t '
              f'model name: {m.name}, settings: {settings}')
        gen_model_fn(**settings)
        logging.info('training')
        m.train(train_g, val_g, epochs=args.epochs,
                tensorboard=(not args.no_tensorboard), callbacks=callbacks)
        logging.info('evaluating')
        m.eval(test_g, args.class_report)
        m.write_to_file(m.name + '.json', args_repr, settings)
