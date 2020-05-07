import json
from random import choice, randint


def random_seq(n=50):
    return ''.join([choice('ACGT') for _ in range(n)])


for sk in ['Archaea', 'Eukaryota', 'Viruses', 'Bacteria']:
    json.dump([random_seq(randint(10, 50)) for _ in range(randint(5, 20))],
              open(f'{sk}_fragments.json', 'w'), indent=2)
