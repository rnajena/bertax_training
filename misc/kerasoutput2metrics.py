import re
import numpy as np
import matplotlib.pyplot as plt


def slurm2metrics(slurmfile, tillepoch=None):
    metrics = []
    with open(slurmfile) as f:
        base = r'^.* - (loss): ([\d\.]+) - (MLM_loss): ([\d\.]+) - (NSP_loss): ([\d\.]+)'
        withval = base + r'(?: - (val_loss): ([\d\.]+) - (val_MLM_loss): ([\d\.]+) - (val_NSP_loss): ([\d\.]+))?'
        for line in f.readlines():
            if (tillepoch is not None and line.startswith(f'Epoch {tillepoch}/')):
                break
            if not ' - loss: ' in line:
                continue
            match = re.match(withval, line)
            metrics.append({match[i]: float(match[i+1]) for i in range(1, match.lastindex, 2)})
    return metrics

bert_nc_metrics = slurm2metrics('slurm-370261.out', 14)
bert_v0_metrics = slurm2metrics('slurm-322628.out')
keys = ['loss', 'MLM_loss', 'NSP_loss', 'val_loss', 'val_MLM_loss', 'val_NSP_loss']
keys_min = ['loss', 'val_loss']
keys = keys_min


def plot_metrics(metrics, name):
    cond = lambda key, m: key in m
    plt.clf()
    for key in keys:
        plt.plot([i for i, m in enumerate(metrics) if cond(key, m)],
                 [m[key] for m in metrics if cond(key, m)])
    plt.legend(keys)
    plt.savefig(name + '.png')


plot_metrics(bert_v0_metrics, 'bert_v0')
plot_metrics(bert_nc_metrics, 'bert_nc', 2)
