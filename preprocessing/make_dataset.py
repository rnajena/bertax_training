from models.bert_nc_finetune import load_fragments
from models.bert_utils import get_classes_and_weights_multi_tax
import argparse
import pandas as pd
import itertools
from utils.tax_entry import TaxidLineage
import time
import numpy as np
import os
import pickle

def choose_sub_class_to_cut_out(tax_list, upper_rank, lower_rank):
    # e.g {phylum: {class1: [taxid1,...], class2: []}
    lineage_list = [tlineage.get_ranks(i, ranks=[upper_rank, lower_rank]) for i in tax_list]

    interest_dict = {}

    for tax_entry in lineage_list:
        upper_taxid, upper = tax_entry[upper_rank]
        lower_taxid, lower = tax_entry[lower_rank]
        if upper_taxid is None or lower_taxid is None:
            continue
        if not upper in interest_dict.keys():
            interest_dict.update({upper: {}})
        if not lower in interest_dict[upper].keys():
            interest_dict[upper].update({lower: []})
        interest_dict[upper][lower].append(lower_taxid)

    upper_to_chosen_ids = {}
    threshold = 2000
    # max_diff = 0
    min_sum = 2000
    best_sum = 0
    # print("dict_len", interest_dict.keys())
    for upper in sorted(interest_dict.keys()):
        lowers = interest_dict[upper]
        min_diff_to_threshold = 1e10
        best_combo = None
        start = time.time()
        lowers_len = {c: len(lowers[c]) for c in lowers.keys()}
        finish=False
        for r in range(1, 5):
            for combo in itertools.combinations(lowers, r):
                s = sum(lowers_len[c] for c in combo)  # s = sum(len(lowers[c]) for c in combo)
                diff = abs(threshold - s)
                if diff < min_diff_to_threshold:
                    min_diff_to_threshold = diff
                    best_combo = combo
                    best_sum = s
                    # if diff == 0:
                    #     break
                    if min_sum <= best_sum <= 2000:
                        finish=True
                        break
            if finish:
                break

        if min_sum > best_sum:
            min_sum = best_sum
        end = time.time()
        print(end - start)

        print(best_combo, best_sum)
        upper_to_chosen_ids.update({upper: [x for c in best_combo for x in lowers[c]]})
    print("dict_len", upper_to_chosen_ids.keys())

    return upper_to_chosen_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='make dataset for multiple architecture testing')
    parser.add_argument('fragments_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--nr_seqs', help=' ', type=int, default=None)
    parser.add_argument('--unbalanced', help=' ', action='store_true')
    parser.add_argument('--filtered', help=' ', action='store_true')
    parser.add_argument('--no_unknown', help=' ', action='store_true')
    parser.add_argument('--num_min_train_samples', help=' ', type=int, default=10000)
    parser.add_argument('--num_test_samples', help=' ', type=int, default=2000)

    tlineage = TaxidLineage()
    args = parser.parse_args()

    assert args.unbalanced == (args.nr_seqs == None), "either use nr_seqs or unbalanced parameter"

    x, y, y_species = load_fragments(args.fragments_dir, nr_seqs=args.nr_seqs, balance=not args.unbalanced)
    # parent_dict, scientific_names, common_names, phylo_names, genbank_common_name, scientific_names_inv, common_names_inv = get_dicts()
    classes, weight_classes, species_list_y = get_classes_and_weights_multi_tax(y_species,
                                                                                tax_ranks=['superkingdom', 'phylum'], unknown_thr=args.num_test_samples+args.num_min_train_samples)
    # tax_ranks=['superkingdom', 'phylum'])


    dir = args.out_dir
    # dir = "/home/go96bix/projects/dna_class/resources/"
    assert os.path.isfile(dir)==False, f"{dir} is a file, please set a directory as out_dir"
    if os.path.isdir(dir):
        pass
    else:
        os.makedirs(dir)

    if args.no_unknown:
        mask = species_list_y[:, 1] != "unknown"
        x = x[mask]
        y = y[mask]
        y_species = y_species[mask]
        species_list_y = species_list_y[mask]

    if args.filtered:
        dir += "filtered/"
        ids_to_filter = choose_sub_class_to_cut_out(y_species, "phylum", "genus")
        ids_to_filter_list = np.unique(np.array([j for i in ids_to_filter.values() for j in i]))
        num_samples_to_draw = min([len(i) for i in ids_to_filter.values()])
        # mask_test = [i in ids_to_filter_list for i in y_species]
        mask_test = [tlineage.get_ranks(i, ranks=['genus'])['genus'][0] in ids_to_filter_list for i in y_species]
        x_test = x[mask_test]
        y_test = y[mask_test]
        y_species_test = y_species[mask_test]
        species_list_y_test = species_list_y[mask_test]
        df_test = pd.DataFrame({"x": x_test, "y": y_test, "tax_id": y_species_test,
                                "superkingdom": species_list_y_test[:, 0], "phylum": species_list_y_test[:, 1]})
        df_test = df_test.groupby("phylum").sample(n=num_samples_to_draw, random_state=1)

        not_mask_test = np.logical_not(mask_test)
        x = x[not_mask_test]
        y = y[not_mask_test]
        y_species = y_species[not_mask_test]
        species_list_y = species_list_y[not_mask_test]
        df_train = pd.DataFrame(
            {"x": x, "y": y, "tax_id": y_species, "superkingdom": species_list_y[:, 0], "phylum": species_list_y[:, 1]})

        df_test.to_csv(dir + "test.tsv", sep="\t", index=False)
        df_train.to_csv(dir + "train.tsv", sep="\t", index=False)

    else:
        # pass
        df = pd.DataFrame(
            {"x": x, "y": y, "tax_id": y_species, "superkingdom": species_list_y[:, 0], "phylum": species_list_y[:, 1]})
        df_test = df.groupby("phylum").sample(n=2000, random_state=1)
        test_ids = df_test.index
        df_train = df.drop(test_ids)
        df_test.to_csv(dir + "test.tsv", sep="\t", index=False)
        df_train.to_csv(dir + "train.tsv", sep="\t", index=False)

    pickle.dump(classes,open(os.path.join(dir,'classes.pkl'),'wb'))