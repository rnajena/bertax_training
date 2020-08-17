from misc.conf_matrix import conf_matrix
from misc.metrics import plot_roc, compute_roc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqIO import parse
from re import sub, findall
from os.path import splitext, basename
from tqdm import tqdm


# NOTE: also included in evaluation.ipynb
def bar_plot_classes(loaded):
    y_sum = np.sum(loaded['preds'], axis=0)
    y_max = np.sum([[1 if v == np.argmax(yi) else 0
                     for v in range(len(yi))] for yi in loaded['preds']], axis=0)
    width = 0.4
    for i, y in enumerate([y_sum, y_max]):
        x = [_ + (i * width) for _ in range(len(loaded['classes']))]
        plt.bar(x, y, width=0.4)
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 10, f'{yi/len(loaded["preds"])*100:.0f}%',
                     horizontalalignment='center')
    plt.xticks([_ + width / 2 for _ in range(len(loaded['classes']))],
               loaded['classes'])
    plt.ylim(top=plt.ylim()[1] + 20)
    plt.tight_layout()
    plt.legend(['sum', 'sum(argmax)'])


def relate_eve_preds(loaded, fa, gff):
    fasta = fa
    gff = gff
    records = list(parse(fasta, 'fasta'))
    for xi, yi in zip(loaded['x'], loaded['preds']):
        for ri in [r for r in records if str(r.seq) == xi]:
            ri.pred = ','.join([f'{c}:{val:.2f}'
                                for val, c in zip(yi, loaded['classes'])])
    with open(gff) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            l_loc = (line[0], str(int(line[3]) - 1), line[4], line[6])
            for r in records:
                if ((r.id.split(':')[2], *r.id.split(':')[3][:-3].split('-'),
                     r.id[-2]) == l_loc):
                    r.gene_id = sub(r'.*gene_id "([^\"]+)".*', r'\1',
                                    ''.join(line))
    with open(splitext(basename(fasta))[0] + '_predictions.txt', 'w') as f:
        for r in records:
            f.write(f'{r.gene_id}\t{r.pred}\n')


def relate_eve_preds_nox(loaded, fasta, gff):
    records = list(parse(fasta, 'fasta'))
    for ri, yi in zip(records, loaded['preds']):
        ri.pred = ','.join([f'{c}:{val:.2f}'
                            for val, c in zip(yi, loaded['classes'])])
    with open(gff) as f:
        for line, r in tqdm(zip(f.readlines(), records)):
            line = line.strip().split('\t')
            l_loc = (line[0], str(int(line[3]) - 1), line[4], line[6])
            if ((r.id.split(':')[2], *r.id.split(':')[3][:-3].split('-'),
                 r.id[-2]) == l_loc):
                r.gene_id = sub(r'.*gene_id "([^\"]+)".*', r'\1',
                                ''.join(line))
    with open(splitext(basename(fasta))[0] + '_predictions.txt', 'w') as f:
        for r in records:
            f.write(f'{r.gene_id}\t{r.pred}\n')
    return records


# NOTE: also included in evaluation.ipynb
def plot_conf_matrix(loaded, class_=None, perc=True, **cm_args):
    preds = list(map(loaded['classes'].__getitem__,
                     map(np.argmax, loaded['preds'])))
    if class_ is not None:
        trues = [class_] * len(preds)
    else:
        trues = list(map(loaded['classes'].__getitem__, loaded['y']))
    min_len = min(len(trues), len(preds))
    return conf_matrix(trues[:min_len], preds[:min_len], perc=perc, **cm_args)


# NOTE: also included in evaluation.ipynb
def add_y_text(x, y, y_offset=0.02, prec=3):
    if isinstance(x[0], str):
        x = list(range(len(x)))
    for xi, yi in zip(x, y):
        plt.text(xi, yi + y_offset, f'{yi:.{prec}f}',
                 horizontalalignment='center')
    plt.ylim(top=plt.ylim()[1] + y_offset)


if __name__ == '__main__':
    # # EVEs
    # eves = pickle.load(
    #     open('output/predictions/bert_nc_C2_ep07_finetuned_eve_predictions.pkl', 'rb'))
    # bar_plot_classes(eves)
    # # plt.show()
    # plt.savefig('eve_preds_all.svg')
    # relate_eve_preds(
    #     eves, '/home/fleming/tmp/eves_hsa_all_dbs_1e-10_unamb_strand_only.fa',
    #     '/home/fleming/tmp/viss_in_hsa_1E-10_unamb_strand_only.gtf_7')
    # # EVEs new
    # eves = pickle.load(
    #     open('output/predictions/eve_final_predictions_slim.pkl', 'rb'))
    # r_eve = relate_eve_preds(
    #     eves, 'resources/viss_in_hsa_1E-10.gtf_all_bothstrands.fa',
    #     'resources/viss_in_hsa_1E-10.gtf_all_bothstrands')
    # r_eve_unamb = [r for r in r_eve if not r.gene_id[-1] in ['+', '-']]
    # preds_unamb = np.array([list(map(
    #     float, findall(r'[\d\.]+', r.pred))) for r in r_eve_unamb])
    # EVEs new new
    eves = pickle.load(
        open('output/predictions/eve_predictions_new_slim.pkl', 'rb'))
    r_eve = relate_eve_preds_nox(
        eves, 'output/predictions/viss_in_hsa_1E-10.gtf_all_new_bothstrands.fa',
        'output/predictions/viss_in_hsa_1E-10.gtf_all_new_bothstrands')
    r_eve_unamb = [r for r in r_eve if not r.gene_id[-1] in ['+', '-']]
    preds_unamb = np.array([list(map(
        float, findall(r'[\d\.]+', r.pred))) for r in r_eve_unamb])
    # # genes
    # genes = pickle.load(open(
    #     'output/predictions/bert_nc_C2_ep07_finetuned_10k_genes_predictions.pkl', 'rb'))
    # plot_conf_matrix(genes)
    # plt.savefig('nc_gene_preds.svg')
    # # leave out 10k clades
    # from models.model import PARAMS
    # classes = PARAMS['data']['classes'][1]
    # lo = {c: pickle.load(open(f'output/predictions/10k_clades_{c}_predictions.pkl', 'rb'))
    #       for c in classes}
    # # individual CMs
    # for c in classes:
    #     plot_conf_matrix(lo[c], c)
    #     plt.savefig('10kclades_' + c + '.svg')
    # # combined CMs
    # assert all(lo[c]['classes'] == lo['Viruses']['classes'] for c in classes)
    # lo_combined = {'classes': lo['Viruses']['classes'], 'preds': [], 'y': []}
    # for c in classes:
    #     lo_combined['preds'].extend(lo[c]['preds'])
    #     lo_combined['y'].extend([lo_combined['classes'].index(c)]
    #                             * len(lo[c]['preds']))
    # assert len(lo_combined['preds']) == len(lo_combined['y'])
    # plot_conf_matrix(lo_combined)
    # plt.savefig('10kclades.svg')
    # # -> not too useful
    # # bar plot
    # cms = {c: plot_conf_matrix(lo[c], c, perc=False, rounded=False)
    #        / len(lo[c]['preds']) for c in classes}
    # base_acc = 0.8640351295471191 # from /ssh:ara:/home/lo63tor/slurm/filter_10k_clades:98224
    # x = ['base'] + classes
    # y = [base_acc] + [float(cms[c][c]) for c in classes]
    # plt.clf()
    # plt.bar(x, y, color=['C1'] + ['C0'] * len(classes))
    # add_y_text(x, y, 0, prec=2)
    # plt.axhline(base_acc, ls='--', c='C1')
    # fig = plt.gcf()
    # size = fig.get_size_inches()
    # fig.set_size_inches([size[0] + 2, size[1]])
    # # plt.show()
    # plt.savefig('10kclades_bar.svg')
    # # genes ROC
    # from sklearn.metrics import roc_auc_score
    # roc_auc_score(genes['y'][:len(genes['preds'])], genes['preds'],
    #               multi_class='ovo')
    # from sklearn.metrics import roc_curve, auc
    # from sklearn.preprocessing import label_binarize
    # from scipy import interp
    # from sklearn.metrics import roc_auc_score
    # plot_roc(genes['y'], genes['preds'], genes['classes'])
    # roc = compute_roc(genes['y'], genes['preds'], genes['classes'])
    # from sklearn.metrics import accuracy_score
    # y = label_binarize(genes['y'][:len(genes['preds'])], classes=range(len(genes['classes'])))
    # accuracy_score(genes['y'][:len(genes['preds'])], list(map(np.argmax, genes['preds'])))

    # import tensorflow as tf
    # m = tf.keras.metrics.CategoricalAccuracy()
    # m.update_state(y, genes['preds'])
    # m = tf.keras.losses.CategoricalCrossentropy()
    # m(y, genes['preds']).numpy()
    # from misc.visuals import accuracy, loss
    # print(loss(label_binarize(genes['y'][:len(genes['preds'])],
    #                         classes=range(len(genes['classes']))),
    #                genes['preds']))
    # m.result().numpy()
