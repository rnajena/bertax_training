from models.bert_nc_finetune import load_fragments,get_classes_and_weights_multi_tax
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='make dataset for multiple architecture testing')
    parser.add_argument('fragments_dir')
    parser.add_argument('--nr_seqs', help=' ', type=int, default=None)
    parser.add_argument('--unbalanced', help=' ', action='store_true')

    args = parser.parse_args()

    assert args.unbalanced == (args.nr_seqs == None), "either use nr_seqs or unbalanced parameter"

    x, y, y_species = load_fragments(args.fragments_dir, nr_seqs=args.nr_seqs, balance=not args.unbalanced)
    classes, weight_classes, species_list_y = get_classes_and_weights_multi_tax(y_species,
                                                                                tax_ranks=['superkingdom', 'phylum'])


    dir = "/home/go96bix/projects/dna_class/resources/"
    mask = species_list_y[:, 1] != "unknown"
    x = x[mask]
    y = y[mask]
    y_species = y_species[mask]
    species_list_y = species_list_y[mask]
    df = pd.DataFrame(
        {"x": x, "y": y, "tax_id": y_species, "superkingdom": species_list_y[:, 0], "phylum": species_list_y[:, 1]})
    df_test = df.groupby("phylum").sample(n=2000, random_state=1)
    test_ids = df_test.index
    df_train = df.drop(test_ids)
    df_test.to_csv(dir + "test.tsv", sep="\t", index=False)
    df_train.to_csv(dir + "train.tsv", sep="\t", index=False)