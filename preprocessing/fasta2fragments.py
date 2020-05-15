import json
import argparse
from Bio.SeqIO import parse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta')
    parser.add_argument('out_prefix')
    parser.add_argument('--no_species_txt', action='store_true')
    args = parser.parse_args()
    records = list(parse(args.fasta, 'fasta'))
    fragments = [str(r.seq) for r in records]
    species_list = [r.id for r in records]
    json.dump(fragments, open(args.out_prefix + '_fragments.json', 'w'))
    if (not args.no_species_txt):
        with open(args.out_prefix + '_species_picked.txt', 'w') as f:
            f.write('\n'.join(species_list))
