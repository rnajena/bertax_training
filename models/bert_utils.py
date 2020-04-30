import keras
import keras_bert
from preprocessing.process_inputs import seq2kmers, ALPHABET
from random import randint
import numpy as np
from itertools import product

# NOTE: uses just keras (instead of tensorflow.keras). Otherwise
# loading of the pre-trained model is likely to fail


def get_token_dict(alph=ALPHABET, k=3):
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict


def generate_bert_with_pretrained(pretrained_path, nr_classes=4):
    """get model ready for fine-tuning and the maximum input length"""
    # see https://colab.research.google.com/github/CyberZHG/keras-bert
    # /blob/master/demo/tune/keras_bert_classification_tpu.ipynb
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(pretrained_path, compile=False,
                                    custom_objects=custom_objects)
    inputs = model.inputs[:2]
    nsp_dense_layer = model.get_layer(name='NSP-Dense').output
    softmax_layer = keras.layers.Dense(nr_classes, activation='softmax')(
        nsp_dense_layer)
    model_fine = keras.Model(inputs=inputs, outputs=softmax_layer)
    return model_fine, inputs[0].shape[1]


def seq2tokens(seq, token_dict, max_length=250,
               k=3, stride=3, window=True):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT
    NOTE: intended to be used as `custom_encode_sequence` argument for
    DataGenerators"""
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - max_length - 1, 0))
        end = start + max_length - 1
    else:
        start = 0
        end = len(seq)
    indices = [token_dict['[CLS]']] + [token_dict[word]
                                       if word in token_dict
                                       else token_dict['[UNK]']
                                       for word in seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]


def process_bert_tokens_batch(batch_x):
    """when `seq2tokens` is used as `custom_encode_sequence`, batches
    are generated as [[input1, input2], [input1, input2], ...]. In
    order to train, they have to be transformed to [input1s,
    input2s] with this function"""
    return [np.array([_[0] for _ in batch_x]),
            np.array([_[1] for _ in batch_x])]
