import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fragments_json')
    parser.add_argument('species_txt')
    args = parser.parse_args()
    fragments = json.load(open(args.fragments_json))
    species_list = [line.strip() for line in
                    open(args.species_txt).readlines()]
    assert len(fragments) == len(species_list)
    for fragment, species in zip(fragments, species_list):
        print(f'>{species}\n{fragment}')
