from models.bert_nc_finetune import load_fragments, FragmentGenerator
from preprocessing.generate_data import PredictGenerator, BatchGenerator, DataSplit
from preprocessing.process_inputs import words2index
from models.model import PARAMS
import numpy as np
from sklearn.preprocessing import label_binarize
from random import randint

classes = PARAMS['data']['classes'][1]


def test_batch_generator():
    split = DataSplit('output/test_files', 100, classes, 'output/test_files.json', train_test_split=0, val_split=0)
    g = BatchGenerator(*split.get_train_files(), classes, 4, max_seq_len=20, fixed_size_method='window', enc_method=words2index)
    p_g = PredictGenerator(g, True)
    preds = label_binarize([[p_g[i], randint(0, len(classes))][1]
                            for i in range(len(p_g))], range(len(classes)))
    preds = np.concatenate([preds] * 4)
    p_g.get_targets()
    p_g.get_x()


def test_fragment_generator():
    x, y = load_fragments('test/artificial_test_fragments/')
    f = FragmentGenerator(x, y, 5, max_seq_len=10, k=3, stride=1,
                          batch_size=5, window=True)
    fp = PredictGenerator(f, True)
    preds = label_binarize([[fp[i], randint(0, len(classes))][1]
                            for i in range(len(fp))], range(len(classes)))
    preds = np.concatenate([preds] * 4)
    fp.get_targets()
    fp.get_x()
