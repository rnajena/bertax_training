import pandas as pd

if __name__ == '__main__':

    dir = "/home/go96bix/projects/dna_class/resources/big_set/"

    df_test = pd.read_csv(dir + "test.tsv", sep="\t", index_col=None)
    df_train = pd.read_csv(dir + "train.tsv", sep="\t", index_col=None)
    for name, df in [("query",df_test), ("db_bert", df_train)]:
        with open(f"/mnt/fass2/projects/fm_read_classification_comparison/{name}.fa","w") as outfile:
            for line in df.itertuples():
                outfile.write(f">{line.tax_id} {line.Index}\n")
                outfile.write(f"{line.x}\n")

