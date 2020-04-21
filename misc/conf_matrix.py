if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from sklearn.metrics import confusion_matrix
from re import match
from models.model import PARAMS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


classes = PARAMS['data']['classes'][1]


def plot_confusion(fastafile):
    y_trues = []
    y_preds = []
    with open(fastafile) as f:
        for line in f.readlines():
            if line.startswith('>'):
                y_true, y_pred = match(r'>([^;]+); predicted: (.*)$', line).groups()
                y_trues.append(y_true)
                y_preds.append(y_pred)

    x = confusion_matrix(y_trues,
                     y_preds, classes)
    df_cm = pd.DataFrame(x, classes, classes)
    # plt.figure(figsize=(10,7))
    plt.clf()
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.savefig(fastafile + '_conf_matrix.png')

if __name__ == '__main__':
    plot_confusion('bert_nc_toy_finetuned_test_fragments_1k.txt_seqs.fasta')
    plot_confusion('bert_nc_toy_finetuned_test_coding_seqs_10k_1500nt.txt_seqs.fasta')
