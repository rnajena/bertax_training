if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
from models.model import PARAMS
import keras
import keras_bert
from preprocessing.generate_data import load_fragments, DataSplit
from preprocessing.process_inputs import read_seq
from preprocessing.process_inputs import seq2kmers, get_class_vectors
from sklearn.metrics import classification_report
from keras.utils import Sequence
from models.bert_nc import get_token_dict
from tqdm import tqdm

classes = PARAMS['data']['classes'][1]
batch_size = 1000
token_dict = get_token_dict()


class FragmentGenerator(Sequence):

    def __init__(self, x, y, seq_len):
        global classes
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.class_vectors = get_class_vectors(classes)

    def __len__(self):
        global batch_size
        return np.ceil(len(self.x) /
                       float(batch_size)).astype(np.int)

    def __getitem__(self, idx):
        global token_dict
        batch_fragments = self.x[idx * batch_size:
                                 (idx + 1) * batch_size]
        batch_classes = self.y[idx * batch_size:
                               (idx + 1) * batch_size]
        batch_y = np.array([self.class_vectors[c] for c in batch_classes])
        batch_seqs = [seq2kmers(seq, k=3, stride=3, pad=False, to_upper=True)
                      for seq in batch_fragments]
        batch_x = []
        for seq in batch_seqs:
            indices = [token_dict['[CLS]']] + [token_dict[word]
                                               if word in token_dict
                                               else token_dict['[UNK]']
                                               for word in seq]
            if (len(indices) < self.seq_len):
                indices += [token_dict['']] * (self.seq_len - len(indices))
            else:
                indices = indices[:self.seq_len]
            segments = [0 for _ in range(self.seq_len)]
            batch_x.append([np.array(indices), np.array(segments)])
        return ([np.array([_[0] for _ in batch_x]),
                np.array([_[1] for _ in batch_x])], batch_y)


def load_bert_model(model_path):
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    return keras.models.load_model(model_path,
                                   custom_objects=custom_objects,
                                   compile=False)


def test_pregen_fragments(fragments_dir, nr_seqs=50_000, outfile='tmp.txt', saveseqs=True):
    x, y = load_fragments(fragments_dir, classes, nr_seqs=nr_seqs)
    model = load_bert_model('output/bert_nc_finetuned.h5')
    y_pred = model.predict(FragmentGenerator(x, y, 502), verbose=1)
    y_pred = np.argmax(y_pred, 1)
    y_true = [classes.index(_) for _ in y]
    c_report = classification_report(y_true, y_pred,
                                     target_names=classes)
    print(c_report)
    with open(outfile, 'w') as f:
        f.write(c_report)
    if (saveseqs):
        with open(outfile + '_seqs.fasta', 'w') as f:
            for xi, yi, ypredi in zip(x, y_true, y_pred):
                f.write(f'>{classes[yi]}; predicted: {classes[ypredi]}\n{xi}\n')


def test_coding_seqs(nr_seqs=10_000, orig_seq_len_nt=None,
                     seq_len=502, outfile='bert_nc_toy_finetuned_test_coding_seqs_report.txt',
                     saveseqs=False,
                     root_fa_dir=PARAMS['data']['root_fa_dir'][1],
                     file_names_cache=PARAMS['data']['file_names_cache'][1]):
    split = DataSplit(root_fa_dir,
                      (nr_seqs if orig_seq_len_nt is None else 250_000),
                      classes, file_names_cache,
                      train_test_split=0, val_split=0)
    files, y = split.get_train_files()
    print('reading in files')
    if (orig_seq_len_nt is None):
        x = [read_seq(_) for _ in tqdm(files)]
    else:
        x = []
        y_new = []
        pbars = {c: tqdm(total=nr_seqs, position=i,
                         desc=c)
                 for i, c in enumerate(classes)}
        its = 0
        for f, label in zip(files, y):
            if (len(x) >= (nr_seqs * len(classes))):
                break
            if (pbars[label].n >= nr_seqs):
                continue
            seq = read_seq(f)
            its += 1
            if (len(seq) >= orig_seq_len_nt):
                x.append(seq)
                y_new.append(label)
                pbars[label].update()
        else:
            raise Exception('not enough sequences with enough seq_len!')
        y = y_new
        print(f'needed to read {its} seqs for the desired {len(x)}')
    model = load_bert_model('output/bert_nc_finetuned.h5')
    g = FragmentGenerator(x, y, seq_len)
    y_pred = model.predict_generator(g,
                                     verbose=1)
    y_pred = np.argmax(y_pred, 1)
    y_true = [classes.index(_) for _ in y]
    c_report = classification_report(y_true, y_pred,
                                     target_names=classes)
    print(c_report)
    with open(outfile, 'w') as f:
        f.write(c_report)
    if (saveseqs):
        with open(outfile + '_seqs.fasta', 'w') as f:
            for xi, yi, ypredi in zip(x, y_true, y_pred):
                f.write(f'>{classes[yi]}; predicted: {classes[ypredi]}\n{xi}\n')


if __name__ == '__main__':
    # test_coding_seqs(10_000, 1500, outfile='bert_nc_toy_finetuned_test_coding_seqs_10k_1500nt.txt', saveseqs=True)
    test_pregen_fragments('output/genomic_fragments', 1_000, 'bert_nc_toy_finetuned_test_fragments_1k.txt')
