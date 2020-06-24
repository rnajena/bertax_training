		   _________________________________

		    CLASSIFICATION OF DNA SEQUENCES
		   _________________________________


Table of Contents
_________________

1. Modules
.. 1. `models'
.. 2. `preprocessing'
.. 3. `utils'
2. Usage
.. 1. Training models
..... 1. Data preparation
..... 2. Training and testing models
.. 2. Using BERT models
3. Dependencies


Library for classifying DNA Sequences.

Written to classify sequences by taxonomy but easily adaptable for other
classification tasks.


1 Modules
=========

1.1 `models'
~~~~~~~~~~~~

  Contains definitions of model architectures based on convolutional,
  recurrent, and temporal convolutional neural networks as well as
  scripts for pre-training and fine-tuning BERT models adapted for
  protein-coding gene sequences and genomic sequences


1.2 `preprocessing'
~~~~~~~~~~~~~~~~~~~

  Contains utilities for sequence input, encoding and model input
  generation.


1.3 `utils'
~~~~~~~~~~~

  Contains various scripts for training and evaluating model
  architectures as well as predicting data using these models.


2 Usage
=======

2.1 Training models
~~~~~~~~~~~~~~~~~~~

2.1.1 Data preparation
----------------------

  Two modes exist for preparing raw DNA sequences for training


* 2.1.1.1 Individual sequence fastas (`gene')

  Each sequence is contained in a fasta file, additionally, a `json'
  file containg all file-names and associated classes can speed up
  preprocessing tremendously. A fixed directory structure is requiered:
  ,----
  | [class_1]/
  |   [sequence_1.fa]
  |   [seuqence_2.fa]
  |   ...
  |   [sequence_n.fa]
  | [class_2]/
  |   ...
  | .../
  | [class_n]/
  |   ...
  |   [sequence_l.fa]
  | {files.json}
  `----

  The `json'-files cotains a list of two lists with equal size, the
  first list contains filepaths to the fasta files and the second list
  the associated classes:
  ,----
  | [["class_1/sequence1.fa", "class_1/sequence2.fa", ..., "class_n/sequence_l.fa"],
  |  ["class_1", "class_1", ..., "class_n"]]
  `----


* 2.1.1.2 single sequence `json'

  This mode requires a fixed directory structure with one `json' file
  per class, consisting of a simple list of sequences:
  ,----
  | [class_1]/
  |   [class_1]_fragments.json
  | [class_2]/
  |   [class_2]_fragments.json
  | .../
  | [class_n]/
  |   [class_n]_fragments.json
  `----

  Example fragments file:
  ,----
  | ["ACGTACGTACGATCGA", "TACACTTTTTA", ..., "ATACTATCTATCTA"]
  `----


2.1.2 Training and testing models
---------------------------------

  All Scripts described here are implemented as CLIs; detailed usage
  information can be optained via the `--help' flag.

  For RNN, CNN and TCN models, the script <file:utils/test_model.py> is
  used:
  ,----
  | python utils/test_model.py tcn --nr_seqs 10_000 --summary \
  |        --root_fa_dir sequences --file_names_cache sequences/files.json
  `----

  To pre-train BERT (gene) models (Script
  <file:models/bert_pretrain.py>):
  ,----
  | python -m models.bert_pretrain bert_gene_C2 --epochs 10 --batch_size 32 --seq_len 502 \
  |        --head_num 5 --embed_dim 250 --feed_forward_dim 1024 --dropout_rate 0.05 \
  |        --root_fa_dir sequences --from_cache sequences/files.json
  `----

  To fine-tune BERT (genomic) models (Script
  <file:models/bert_finetune.py>)
  ,----
  | python -m models.bert_finetune bert_gene_C2_trained.h5 --epochs 4 \
  |        --root_fa_dir sequences --from_cache sequences/files.json
  `----

  The scripts <file:models/bert_nc.py> and
  <file:models/bert_nc_finetune.py> are used analogously, with the
  exception of sequence specification:

  ,----
  | python -m models.bert_nc single_sequence_json_folder/
  `----

  ,----
  | python -m models.bert_nc_finetune bert_nc_trained.h5 single_sequence_json_folder/
  `----


2.2 Using BERT models
~~~~~~~~~~~~~~~~~~~~~

  A script is available to predict sequences in using a BERT model.  For
  example, sequences contained in a fasta file can be predicted:

  ,----
  | > class_1
  | ACGTAGCTA
  | > class_2
  | ACATATATTATATTTT
  `----

  ,----
  | python -m utils.test_bert finetuned_bert.h5 --fasta sequences.fa
  `----

  The best-performing fine-tuned BERT models ready to use are contained
  in the directory `resources/'.

  For this script `--help' provides further usage information.


3 Dependencies
==============

  - tensorflow >= 2
  - keras
  - numpy
  - tqdm
  - scikit-learn
  - keras-bert
  - keras-tcn
