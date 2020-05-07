import json
import numpy as np
from models.model import PARAMS


def filter(files, **kwargs):
    def pick_parent(j, k):
        for cand in ['specific_params', 'params']:
            if (cand in j and k in j[cand]):
                return j[cand]
        if (k in j):
            return j
        return None
    names = []
    for name in files:
        j = json.load(open(name))
        for k, v in kwargs.items():
            parent = pick_parent(j, k)
            if (parent is None):
                break
            if (not parent[k] == v):
                break
        else:
            if ('test_metrics' in j and 'acc' in j['test_metrics']):
                acc = np.round(j['test_metrics']['acc'], 2)
            else:
                acc = np.nan
            names.append((name, acc))
    return names


def p_iter(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from p_iter(v)
        else:
            yield((k, v))


PARAMS_flat = {k: v for k, v in p_iter(PARAMS)}


def params(j_dict):
    params = {}
    if ('params' in j_dict):
        params.update(j_dict['params'])
    if ('specific_params' in j_dict):
        params.update(j_dict['specific_params'])
    return params


def nondefaults(params):
    to_ret = {}
    for k, v in params.items():
        if (k not in PARAMS_flat or PARAMS_flat[k][1] != v):
            to_ret[k] = (v, PARAMS_flat[k][1] if k in PARAMS_flat else None)
    return to_ret
