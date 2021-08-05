#+TITLE: BERTax training utilities
#+OPTIONS: ^:nil
This repository contains utilities for pre-training and fine-tuning [[https://github.com/f-kretschmer/bertax][BERTax]] models, as
well as various utility functions and scripts used in the development of BERTax.

Additionally to the [[https://doi.org/10.1101/2021.07.09.451778][described mode of training BERTax on /genomic/ DNA sequences]],
development scripts for training on /gene/ sequences are included as well.

* Training new BERTax models
** Data preparation
For the training of a new BERTax model the user must provide one of the two following data-structures.

*** fragments directories

For training the normal, genomic DNA-based models, a fixed directory structure with one =json= file
per class, consisting of a simple list of sequences is required:
#+begin_example
[class_1]/
  [class_1]_fragments.json
[class_2]/
  [class_2]_fragments.json
.../
[class_n]/
  [class_n]_fragments.json
#+end_example

Example fragments file:
#+begin_src json
["ACGTACGTACGATCGA", "TACACTTTTTA", ..., "ATACTATCTATCTA"]
#+end_src
*** gene model training directories

The gene models were used in an early stage of BERTax development, where a different
directory structure was required:

Each sequence is contained in a fasta file, additionally, a =json=
file containg all file-names and associated classes can speed up
preprocessing tremendously.

#+begin_example
[class_1]/
  [sequence_1.fa]
  [seuqence_2.fa]
  ...
  [sequence_n.fa]
[class_2]/
  ...
.../
[class_n]/
  ...
  [sequence_l.fa]
{files.json}
#+end_example

The =json=-files cotains a list of two lists with equal size, the
first list contains filepaths to the fasta files and the second list
the associated classes:
#+begin_src json
[["class_1/sequence1.fa", "class_1/sequence2.fa", ..., "class_n/sequence_l.fa"],
 ["class_1", "class_1", ..., "class_n"]]
#+end_src
** Training process
The normal, genomic DNA-based model can be pre-trained with [[file:models/bert_nc.py]] and
fine-tuned with [[file:models/bert_nc_finetune.py]].

For example, the BERTax model was pre-trained with:
#+begin_src shell
  python -m models.bert_nc fragments_root_dir --batch_size 32 --head_num 5 \
         --transformer_num 12 --embed_dim 250 --feed_forward_dim 1024 --dropout_rate 0.05 \
         --name bert_nc_C2 --epochs 10
#+end_src

and fine-tuned with:
#+begin_src shell
  python -m models.bert_nc_finetune bert_nc_C2.h5 fragments_root_dir --multi_tax \
         --epochs 15 --batch_size 24 --save_name _small_trainingset_filtered_fix_classes_selection \
         --store_predictions --nr_seqs 1000000000
#+end_src

The development gene models can be pre-trained with [[file:models/bert_pretrain.py]]:
#+begin_src shell
  python -m models.bert_pretrain bert_gene_C2 --epochs 10 --batch_size 32 --seq_len 502 \
	 --head_num 5 --embed_dim 250 --feed_forward_dim 1024 --dropout_rate 0.05 \
	 --root_fa_dir sequences --from_cache sequences/files.json
#+end_src

and fine-tuned with [[file:models/bert_finetune.py]]:
#+begin_src shell
  python -m models.bert_finetune bert_gene_C2_trained.h5 --epochs 4 \
	 --root_fa_dir sequences --from_cache sequences/files.json
#+end_src

All training scripts can be called with the =--help= flag to adjust various parameters.

** Using BERT models

It is recommended to use fine-tuned models in the BERTax tool with the parameter
=--custom_model_file=.

However, a much more minimal script to predict multi-fasta sequences with the trained
model is also available in this repository:

#+begin_src shell
python -m utils.test_bert finetuned_bert.h5 --fasta sequences.fa
#+end_src
** Benchmarking
If the user needs a predefined training and test set, for example for benchmarking different approaches:

#+begin_src shell
  python -m preprocessing.make_dataset single_sequences_json_folder/ out_folder/ --unbalanced
#+end_src
This creates a the files test.tsv, train.tsv, classes.pkl which can be used by bert_nc_finetune

#+begin_src shell
  python -m models.bert_nc_finetune bert_nc_trained.h5 make_dataset_out_folder/ --unbalanced --use_defined_train_test_set
#+end_src

If fasta files are necessary, e.g., for competing methods, you can parse the train.tsv and test.tsv via
#+begin_src shell
  python -m preprocessing.dataset2fasta make_dataset_out_folder/
#+end_src

* Additional scripts
- [[file:preprocessing/fasta2fragments.py]] / [[file:preprocessing/fragments2fasta.py]] :: convert
  between multi-fasta and json training files
- [[file:preprocessing/genome_db.py]], [[file:preprocessing/genome_mince.py]] :: scripts used to
  generate genomic fragments for training

* Dependencies
- tensorflow >= 2
- keras
- numpy
- tqdm
- scikit-learn
- keras-bert
- biopython
