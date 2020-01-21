import numpy as np
import matplotlib.pyplot as plt


def opt_split(n, min_, max_):
    min_x = n // max_
    max_x = n // min_
    if (max_x <= 2):
        return 2
    if (min_x % 2 == 1):
        min_x += 1
    # c is a factor that achieves convergence to (min_ + max_) / 2
    c = (min_ + max_)**2/(2 * min_ * max_)
    x = (max_x + max(min_x, 2)) // c
    if (x % 2 == 0):
        return x
    else:
        if (x + 1 <= max_x):
            return x + 1
        elif (x - 1 >= min_x):
            return x - 1
        elif (min_x - 1 >= 2):
            return min_x - 1
        return 2


def os_simple_x(n, min_, max_):
    return (n // max_ + n // min_) / ((min_ + max_)**2/(2 * min_ * max_))


def test_split_fns(n, min_, max_, *split_fns):
    for split_fn in split_fns:
        print(f'{split_fn.__name__}({n}, {min_}, {max_}) = '
              f'{split_fn(n, min_, max_)}')


def plot_split_fns(ns, min_, max_, *split_fns):
    for split_fn in split_fns:
        plt.plot(np.array(list(map(
            lambda n: n/split_fn(n, min_, max_), ns))), ns)
    # plt.legend([split_fn.__name__ for split_fn in split_fns])
    plt.show()


plot_split_fns(np.arange(500, 100000/2, 1), 50, 250, opt_split)
plt.savefig('/home/lo63tor/master/dna_class/output/opt_split_sim.png', dpi=60)
