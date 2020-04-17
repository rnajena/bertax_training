if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from preprocessing.generate_data import load_fragments
from models.model import PARAMS
classes = PARAMS['data']['classes'][1]


if __name__ == '__main__':
    x, y = load_fragments('output/genomic_fragments', classes, nr_seqs=10)
    with open('misc/random_fragments_10.fasta', 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f'>{yi}\n{xi}\n')
