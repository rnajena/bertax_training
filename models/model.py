# various default settings used across scripts
# {group: {param_name:
#   (type|(list, type), default, model_type, help, choices)}}
PARAMS = {'data':
          # * everything data-related *
          {'data_source':
           (str, 'genes', None,
            'dataset type to use', ['genes', 'fragments']),
           'classes':
           ((list, str),
            ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota']),
           'nr_seqs': (int, 10_000), 'batch_size': (int, 500),
           'fixed_size_method': (
               str, 'pad', None,
               'Method for transforming sequences to fixed length',
               ['pad', 'window', 'repeat']),
           'rev_comp': (bool, False), 'rev_comp_mode': (
               str, 'append', None, '', ['append', 'random',
                                         'independent']),
           'enc_dimension': (int, 65),
           'enc_k': (int, 3),
           'enc_stride': (int, 3),
           'cache_batches': (bool, True),
           'cache_seq_limit': (int, None),
           'root_fa_dir':
           (str, 'sequences'),
           'root_fragments_dir':
           (str, 'fragments'),
           'file_names_cache':
           (str,
            'sequences/files.json'),
           'enc_method':
           (str, 'words2index', None, '',
            ['words2index', 'words2onehot', 'words2vec']),
           'w2vfile': (str, None, None, 'filename of a pickled word '
                       'vector dict'),
           'bert_token_dict_json':
           (str, '', None, 'path to the JSON-serialized keras-bert '
            'token dict'),
           'bert_pretrained_path':
           (str, '', None, 'path to pre-trained keras-bert model'),
           'max_seq_len': (int, 10_000, None,
                           'Length of *all* sequences when '
                           'using any `fixed_size_method`')}}
