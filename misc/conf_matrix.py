from sklearn.metrics import confusion_matrix
from re import match
from models.model import PARAMS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np


classes = PARAMS['data']['classes'][1]


def conf_matrix(trues, preds, dropzero=True, perc=False, rounded=True,
                xrot=None):
    x = confusion_matrix(trues, preds, classes)
    if perc:
        x = np.round(x / len(preds) * 100)
        if rounded:
            x = np.round(x)
    df_cm = pd.DataFrame(x, classes, classes)
    pd.set_option('precision', 0)
    if (dropzero):
        df_cm = df_cm.drop([c for c in classes if c not in trues])
    # plt.figure(figsize=(10,7))
    plt.clf()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},
               cbar_kws={'format': '%.0f'})  # font size
    plt.yticks(rotation=0)
    if xrot is not None:
        plt.xticks(rotation=xrot)
    plt.tight_layout()
    return df_cm


def fasta_cm(fastafile):
    y_trues = []
    y_preds = []
    with open(fastafile) as f:
        for line in f.readlines():
            if line.startswith('>'):
                y_true, y_pred = match(r'>([^;]+); predicted: (.*)$', line
                                       ).groups()
                y_trues.append(y_true)
                y_preds.append(y_pred)
    conf_matrix(y_trues, y_preds)
    plt.savefig(fastafile + '_conf_matrix.png')


if __name__ == '__main__':
    fasta_cm('bert_nc_toy_finetuned_test_fragments_1k.txt_seqs.fasta')
    fasta_cm('bert_nc_toy_finetuned_test_coding_seqs_10k_1500nt.txt_seqs.fasta')
