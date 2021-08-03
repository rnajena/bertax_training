import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='parse dataset to fastas')
    parser.add_argument('dataset', help='directory containing test.tsv, train.tsv, classes.pkl')
    args = parser.parse_args()

    dir = args.dataset

    df_test = pd.read_csv(dir + "test.tsv", sep="\t", index_col=None)
    df_train = pd.read_csv(dir + "train.tsv", sep="\t", index_col=None)
    for name, df in [("query",df_test), ("db_bert", df_train)]:
        with open(f"{os.path.join(dir,name)}.fa","w") as outfile:
            for line in df.itertuples():
                outfile.write(f">{line.tax_id} {line.Index}\n")
                outfile.write(f"{line.x}\n")

