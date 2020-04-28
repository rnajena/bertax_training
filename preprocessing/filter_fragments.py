import json
import argparse
from ete3 import NCBITaxa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+')
    parser.add_argument('taxid', nargs='+', type=int)
    parser.add_argument('--out_prefix', default='filtered_')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    if (len(args.input) == 2):
        fragments = json.load(open(args.input[0]))
        species_list = [int(line.strip()) for line in
                        open(args.input[1]).readlines()]
        assert len(species_list) == len(fragments)
        iterator = zip(fragments, species_list)
        out_mode = 'json'
        print('json + txt ')
    elif (len(args.input) == 1):
        from Bio.SeqIO import parse
        records = parse(open(args.input[0]), 'fasta')
        iterator = ((str(r.seq), int(r.id)) for r in records)
        out_mode = 'fasta'
    else:
        raise Exception(
            'input either has to be a json and a txt-list or a fasta')
    print(('json + txt' if out_mode == 'json' else 'fasta')
          + ' has been provided as input, output will be of the same format')
    ncbi = NCBITaxa()
    species_filter = []
    print('filtering ' + ','.join(ncbi.translate_to_names(args.taxid)))
    for taxid in args.taxid:
        try:
            species_filter.extend(ncbi.get_descendant_taxa(taxid, True))
        except ValueError:
            print('could not get descendant taxa of ' + ','.join(
                ncbi.translate_to_names([taxid]))
                + ', filtering only for this taxon; ')
            species_filter.append(taxid)
    filtered_fragments = []
    filtered_species = []
    for fragment, species in iterator:
        if ((not args.reverse and species not in species_filter) or (
                args.reverse and species in species_filter)):
            filtered_fragments.append(fragment)
            filtered_species.append(species)
    if (out_mode == 'json'):
        json.dump(filtered_fragments, open(
            args.out_prefix + 'fragments.json', 'w'))
        with open(args.out_prefix + 'species.txt', 'w') as f:
            for species in filtered_species:
                f.write(str(species) + '\n')
    else:
        with open(args.out_prefix + '.fa', 'w') as f:
            for fragment, species in zip(filtered_fragments, filtered_species):
                f.write(f'>{species}\n{fragment}\n')
