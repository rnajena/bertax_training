import json
import argparse
from ete3 import NCBITaxa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fragments_json')
    parser.add_argument('species_list')
    parser.add_argument('taxid', nargs='+', type=int)
    parser.add_argument('--out_prefix', default='filtered_')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    fragments = json.load(open(args.fragments_json))
    species_list = [int(line.strip()) for line in
                    open(args.species_list).readlines()]
    assert len(species_list) == len(fragments)

    ncbi = NCBITaxa()
    species_filter = []
    print('filtering ' + ','.join(ncbi.translate_to_names(args.taxid)))
    for taxid in args.taxid:
        species_filter.extend(ncbi.get_descendant_taxa(taxid, True))
    filtered_fragments = []
    filtered_species = []
    for fragment, species in zip(fragments, species_list):
        if ((not args.reverse and species not in species_filter) or (
                args.reverse and species in species_filter)):
            filtered_fragments.append(fragment)
            filtered_species.append(species)
    json.dump(filtered_fragments, open(
        args.out_prefix + 'fragments.json', 'w'))
    with open(args.out_prefix + 'species.txt', 'w') as f:
        for species in filtered_species:
            f.write(str(species) + '\n')
