import argparse
import logging
from itertools import product
import tensorflow as tf
if (tf.__version__.startswith('1.')):
    import keras.callbacks as keras_cbs
else:
    import tensorflow.keras.callbacks as keras_cbs
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import models.model as model
from preprocessing.generate_data import DataSplit, load_fragments, FragmentGenerator
from preprocessing.process_inputs import words2index, words2onehot, words2vec
from sklearn.model_selection import train_test_split


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
                                         'tcn', 'ff', 'bert'],
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
    # validate arguments + argument preprocessing
    # bert-only to_generators-functions
    custom_encode_sequence = None
    process_batch_function = None
    if (args.type == 'bert'):
        from os.path import exists
        if (any([path == '' or not exists(path) for path in
                 [args.bert_token_dict_json, args.bert_pretrained_path]])):
            raise Exception('both `bert_token_dict_json` and '
                            '`bert_pretrained_path` have to be specified and '
                            'be valid files')
        import json
        token_dict = json.load(open(args.bert_token_dict_json))
        logging.info('loaded BERT token dict')
        from models.bert_utils import seq2tokens, process_bert_tokens_batch
        if (args.fixed_size_method not in ['pad', 'window']):
            raise Exception('for bert, only the fixed_size_methods pad and '
                            'window are implemented as of now')
        window = args.fixed_size_method == 'window'
        custom_encode_sequence = (
            lambda seq: seq2tokens(
                seq, token_dict, max_length=args.max_seq_len, window=window,
                k=args.enc_k, stride=args.enc_stride))
        process_batch_function = process_bert_tokens_batch
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
        logging.getLogger().setLevel(logging.INFO)
        from pprint import pprint
        pprint(args)
    # general settings
    logging.info('splitting and balancing dataset')
    if (args.data_source == 'genes'):
        split = DataSplit(args.root_fa_dir, args.nr_seqs,
                          args.classes,
                          from_cache=args.file_names_cache,
                          train_test_split=args.test_split,
                          duplicate_data=(
                              'rev_comp' if (args.rev_comp and
                                             args.rev_comp_mode == 'independent')
                              else None))
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
            w2vfile=args.w2vfile,
            custom_encode_sequence=custom_encode_sequence,
            process_batch_function=process_batch_function)
    elif (args.data_source == 'fragments'):
        x, y = load_fragments(args.root_fragments_dir, args.classes, nr_seqs=args.nr_seqs)
        f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
            x, y, test_size=args.test_split)
        f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
            f_train_x, f_train_y, test_size=0.05)
        generator_params = {
            'seq_len': args.max_seq_len,
            'k': args.enc_k,
            'stride': args.enc_stride,
            'batch_size': args.batch_size,
            'classes': args.classes,
            'fixed_size_method': args.fixed_size_method,
            'enc_method': args.enc_method}
        train_g = FragmentGenerator(f_train_x, f_train_y, **generator_params)
        val_g = FragmentGenerator(f_val_x, f_val_y, **generator_params)
        test_g = FragmentGenerator(f_test_x, f_test_y, **generator_params)
    else:
        raise Exception(f'data source "{args.data_source}" is not accepted')
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
    logging.info(f'creating model architecture: {args.type}')
    gen_model_fn = {'cnn': m.generate_cnn_model,
                    'cnndeep_predef': m.generate_cnndeep_predef_model,
                    'lstm': m.generate_lstm_model,
                    'tcn': m.generate_tcn_model,
                    'ff': m.generate_ff_model,
                    'bert': m.generate_bert_with_pretrained}[args.type]
    if (len(combinations) > 1):
        logging.info('some parameters have multiple values, '
                     'all combinations will be run')
    for i, settings in enumerate(combinations):
        # name = {given_name}_{hash_of_settings}
        import pickle
        from zlib import adler32
        m.name = f'{args.model_name}_{adler32(pickle.dumps(settings))}'
        if (args.model_checkpoints):
            logging.info('enabling model checkpoints')
            if (args.model_checkpoints_keep_all):
                path = m.name + '_cp_ep{epoch:02d}.h5'
            else:
                path = m.name + '_cp.h5'
            callbacks.append(keras_cbs.ModelCheckpoint(path))
        if (args.type == 'tcn'):
            settings['nb_filters'] = settings['nr_filters']
            del settings['nr_filters']
            settings['dilations'] = [2 ** i for i in range(
                settings['last_dilation_2exp'])]
            del settings['last_dilation_2exp']
        elif (args.type == 'bert'):
            settings['pretrained_path'] = args.bert_pretrained_path
        print(f'configuration {i+1}/{len(combinations)}\t '
              f'model name: {m.name}, settings: {settings}')
        gen_model_fn(**settings)
        logging.info('training')
        m.train(train_g, val_g, epochs=args.epochs,
                learning_rate=args.learning_rate,
                tensorboard=(not args.no_tensorboard), callbacks=callbacks)
        logging.info('evaluating')
        m.eval(test_g, args.class_report)
        m.write_to_file(m.name + '.json', args_repr, settings)
