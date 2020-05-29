import re
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename
from itertools import product


def slurm2metrics(slurmfile, tillepoch=None, finetune=False):
    r_pretrain = (r'^.* - (loss): ([\d\.]+) - (MLM_loss): ([\d\.]+) - (NSP_loss): ([\d\.]+)'
                  r'(?: - (val_loss): ([\d\.]+) - (val_MLM_loss): ([\d\.]+) - (val_NSP_loss): ([\d\.]+))?')
    r_finetune = (r'^.* - (loss): ([\d\.]+) - (accuracy): ([\d\.]+)'
                  r'(?: - (val_loss): ([\d\.]+) - (val_accuracy): ([\d\.]+))?')
    r = r_finetune if finetune else r_pretrain
    metrics = []
    with open(slurmfile) as f:
        for line in f.readlines():
            if (tillepoch is not None and line.startswith(f'Epoch {tillepoch}/')):
                break
            if ' - loss: ' not in line:
                continue
            match = re.match(r, line)
            metrics.append({match[i]: float(match[i + 1])
                            for i in range(1, match.lastindex, 2)})
    return metrics


bert_finetune_slurm_files = ['bert_gene_A_finetune', 'bert_gene_A_finetune2', 'bert_gene_D_finetune',
                             'bert_nc_A_finetune', 'bert_nc_C_finetune', 'bert_nc_C2_finetune']
bert_finetune_slurm_files = list(map(lambda x: '/home/fleming/Documents/Projects/dna_class/output/slurm/' + x,
                                     bert_finetune_slurm_files))

bert_nc_A_finetune_metrics = slurm2metrics(
    bert_finetune_slurm_files[3], finetune=True)


def plot_slurms(files, finetune, val_only=True):
    metrics = {basename(f): list(filter(lambda x: not val_only or 'val_loss' in x,
                                        slurm2metrics(f, finetune=finetune)))
               for f in files}
    keys = list(metrics[basename(files[0])][0].keys())
    plt.clf()
    for f in files:
        for k in keys:
            plt.plot([v[k] for v in metrics[basename(f)]])
    plt.legend([f'{basename(f)}: {k}' for (f, k) in product(files, keys)])
    plt.show()


plot_slurms(bert_finetune_slurm_files[:1], True)

bert_nc_metrics = slurm2metrics('slurm-370261.out', 14)
bert_v0_metrics = slurm2metrics('slurm-322628.out')
keys=['loss', 'MLM_loss', 'NSP_loss',
    'val_loss', 'val_MLM_loss', 'val_NSP_loss']
keys_min=['loss', 'val_loss']
keys=keys_min


def plot_metrics(metrics, name):
    cond=lambda key, m: key in m
    plt.clf()
    for key in keys:
        plt.plot([i for i, m in enumerate(metrics) if cond(key, m)],
                 [m[key] for m in metrics if cond(key, m)])
    plt.legend(keys)
    plt.savefig(name + '.png')


plot_metrics(bert_v0_metrics, 'bert_v0')
plot_metrics(bert_nc_metrics, 'bert_nc', 2)
