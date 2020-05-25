from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import tensorflow as tf


def accuracy(trues, preds):
    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(trues, preds)
    return m.result().numpy()


def loss(trues, preds):
    return tf.keras.losses.CategoricalCrossentropy()(trues, preds).numpy()


def compute_roc(trues, preds, classes):
    """computes FPR, TPR, ROC AUC.

    :param trues: either ndarray of shape (n, #classes), list of class labels
                  or list of class indices
    :param preds: ndarray of shape(n, #classes)
    :param classes: list of classes
    """
    min_len = min(len(trues), len(preds))
    if (isinstance(trues, np.ndarray)):
        trues = list(map(np.argmax, trues))
    elif (isinstance(trues[0], str)):
        trues = list(map(classes.index, trues))
    trues = trues[:min_len]
    preds = preds[:min_len]
    y = label_binarize(trues, classes=range(len(classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y[:, i], preds[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), preds.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for c in classes:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    # fpr['macro'] = all_fpr
    # tpr['macro'] = mean_tpr
    # roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    return namedtuple('ROC', 'fpr, tpr, roc_auc')(fpr, tpr, roc_auc)


def plot_roc(trues, preds, classes, all_curves=True):
    fpr, tpr, roc_auc = compute_roc(trues, preds, classes)
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
             label=f'Average (area = {roc_auc["micro"]:.2f})',
             linestyle=(':' if all_curves else '-'), linewidth=4)

    # plt.plot(fpr['macro'], tpr['macro'],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc['macro']),
    #          color='navy', linestyle=':', linewidth=4)
    if (all_curves):
        for c in classes:
            plt.plot(fpr[c], tpr[c],
                     label=f'{c} (area = {roc_auc[c]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if (all_curves):
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return roc_auc
