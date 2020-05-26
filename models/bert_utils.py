import keras
import keras_bert
import tensorflow as tf
from preprocessing.process_inputs import seq2kmers, ALPHABET
from preprocessing.generate_data import PredictGenerator
from random import randint
import numpy as np
from itertools import product
from logging import info
from misc.metrics import compute_roc, accuracy, loss


# NOTE: uses just keras (instead of tensorflow.keras). Otherwise
# loading of the pre-trained model is likely to fail


def get_token_dict(alph=ALPHABET, k=3):
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict


def load_bert(bert_path, compile_=False):
    """get bert model from path"""
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(bert_path, compile=compile_,
                                    custom_objects=custom_objects)
    return model


def generate_bert_with_pretrained(pretrained_path, nr_classes=4):
    """get model ready for fine-tuning and the maximum input length"""
    # see https://colab.research.google.com/github/CyberZHG/keras-bert
    # /blob/master/demo/tune/keras_bert_classification_tpu.ipynb
    model = load_bert(pretrained_path)
    inputs = model.inputs[:2]
    nsp_dense_layer = model.get_layer(name='NSP-Dense').output
    softmax_layer = keras.layers.Dense(nr_classes, activation='softmax')(
        nsp_dense_layer)
    model_fine = keras.Model(inputs=inputs, outputs=softmax_layer)
    return model_fine


def seq2tokens(seq, token_dict, seq_length=250, max_length=None,
               k=3, stride=3, window=True, seq_len_like=None):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT
    NOTE: intended to be used as `custom_encode_sequence` argument for
    DataGenerators"""
    if (max_length is None):
        max_length = seq_length
    if (seq_len_like is not None):
        seq_length = min(max_length, np.random.choice(seq_len_like))
        # open('seq_lens.txt', 'a').write(str(seq_length) + ', ')
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
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


def predict(model, test_generator, roc_auc=True, classes=None,
            return_data=False, nonverbose=False):
    predict_g = PredictGenerator(test_generator)
    preds = model.predict(predict_g, verbose=0 if nonverbose else 1)
    y = predict_g.get_targets()[:len(preds)] # in case not everything was predicted
    acc = accuracy(y, preds)
    loss_ = loss(y, preds)
    if (roc_auc):
        roc_auc = compute_roc(y, preds,
                              classes).roc_auc
        result = [loss_, acc, roc_auc]
        metrics_names = ['test_loss', 'test_accuracy', 'roc_auc']
    else:
        result = [loss_, acc]
        metrics_names = ['test_loss', 'test_accuracy']
    return {'metrics': result, 'metrics_names': metrics_names,
            'data': (y, preds) if return_data else None}
