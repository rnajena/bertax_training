import numpy as np
import keras
import keras_bert
from models.bert_utils import seq2tokens, process_bert_tokens_batch
from models.model import PARAMS
from preprocessing.generate_data import DataSplit
import sys
import json
import pickle

pretrained_path = sys.argv[1]
root_fa_dir = sys.argv[2]
file_names_cache = sys.argv[3]
token_dict = json.load(open(sys.argv[4]))

classes = PARAMS['data']['classes'][1]
batch_size = 100

# mostly copy-pasted from bert_utils.generate_bert_with_pretrained
custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                  'GlorotUniform': keras.initializers.glorot_uniform}
custom_objects.update(keras_bert.get_custom_objects())
model = keras.models.load_model(pretrained_path, compile=False,
                                custom_objects=custom_objects)
inputs = model.inputs[:2]
nsp_dense_layer = model.get_layer(name='NSP-Dense').output
model_vectors = keras.Model(inputs=inputs, outputs=nsp_dense_layer)

split = DataSplit(root_fa_dir=root_fa_dir, nr_seqs=250_000, classes=classes,
                  from_cache=file_names_cache, train_test_split=0,
                  val_split=0, balance=True)
custom_encode_sequence = (
            lambda seq: seq2tokens(
                seq, token_dict, max_length=inputs[0].shape[1], window=True,
                k=3, stride=3))
train_g, val_g, test_g = split.to_generators(
    batch_size=batch_size, custom_encode_sequence=custom_encode_sequence,
    process_batch_function=process_bert_tokens_batch)
predicted = model_vectors.predict(train_g, verbose=1)
pickle.dump(predicted, open('bert_v0_trained_seq_vectors.pkl', 'wb'))
