import tensorflow as tf
from tensorflow import keras
import json
from preprocessing.process_inputs import get_class_vectors, ALPHABET
from models.model import PARAMS
import numpy as np
from tensorflow.keras.utils import Sequence
from models.bert_utils import get_token_dict, seq2tokens, predict
from models.bert_utils import generate_bert_with_pretrained, generate_bert_with_pretrained_multi_tax, get_classes_and_weights_multi_tax
from random import shuffle, sample
from sklearn.model_selection import train_test_split
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from logging import warning
import pickle
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from os.path import splitext
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# /home/go96bix/projects/dna_class/resources/bert_nc_C2_final.h5 /home/go96bix/projects/dna_class/resources/big_set --store_predictions --test_benchmark --multi_tax
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
#
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
mirrored_strategy = tf.distribute.MirroredStrategy()

classes = PARAMS['data']['classes'][1]


def load_dataset(filepath):
    df = pd.read_csv(filepath, sep="\t")
    x = df["x"]
    y = df["y"]
    y_species = df["tax_id"]
    return x, y, y_species


def load_fragments(fragments_dir, shuffle_=True, balance=True, nr_seqs=None):
    fragments = []
    species_list = []
    for class_ in classes:
        fragments.append((class_, json.load(open(os.path.join(
            fragments_dir, f'{class_}_fragments.json')))))
        species_list.append([int(line.strip()) for line in
                             open(os.path.join(fragments_dir, f'{class_}_species_picked.txt')).readlines()])
    nr_seqs_max = min(len(_[1]) for _ in fragments)
    if (nr_seqs is None or nr_seqs > nr_seqs_max):
        nr_seqs = nr_seqs_max
    x = []
    y = np.array([])
    y_species = np.array([], dtype=np.int)

    for index, fragments_i in enumerate(fragments):
        class_, class_fragments = fragments_i
        if not balance:
            x.extend(class_fragments)
            y = np.append(y, [class_] * len(class_fragments))
            y_species = np.append(y_species, species_list[index])
        else:
            x_help = list(zip(class_fragments, species_list[index]))
            # x.extend(sample(class_fragments, nr_seqs))
            x_help = sample(x_help, nr_seqs)
            x_help, y_species_help = zip(*x_help)
            x.extend(x_help)
            y_species = np.append(y_species,y_species_help)
            y = np.append(y, [class_] * nr_seqs)

    assert len(x) == len(y)
    if (shuffle_):
        to_shuffle = list(zip(x, y, y_species))
        shuffle(to_shuffle)
        x, y, y_species = zip(*to_shuffle)
    print(f'{len(x)} fragments loaded in total; '
          f'balanced={balance}, shuffle_={shuffle_}, nr_seqs={nr_seqs}')
    return np.array(x), np.array(y), np.array(y_species)


@dataclass
class FragmentGenerator(Sequence):
    x: list
    y: list
    seq_len: int
    max_seq_len: Optional[int] = None
    k: int = 3
    stride: int = 3
    batch_size: int = 32
    classes: List = field(default_factory=lambda:
    ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'])
    seq_len_like: Optional[np.array] = None
    window: bool = False

    def __post_init__(self):
        self.class_vectors = get_class_vectors(self.classes)
        self.token_dict = get_token_dict(ALPHABET, k=3)
        if (self.max_seq_len is None):
            self.max_seq_len = self.seq_len

    def __len__(self):
        return np.ceil(len(self.x)
                       / float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        batch_fragments = self.x[idx * self.batch_size:
                                 (idx + 1) * self.batch_size]
        batch_x = [seq2tokens(seq, self.token_dict, seq_length=self.seq_len,
                              max_length=self.max_seq_len,
                              k=self.k, stride=self.stride, window=self.window,
                              seq_len_like=self.seq_len_like)
                   for seq in batch_fragments]
        if (self.y is not None and len(self.y) != 0):
            batch_classes = self.y[idx * self.batch_size:
                                   (idx + 1) * self.batch_size]
            batch_y = np.array([self.class_vectors[c] for c in batch_classes])
            return ([np.array([_[0] for _ in batch_x]),
                     np.array([_[1] for _ in batch_x])], [batch_y])
        else:
            return [np.array([_[0] for _ in batch_x]),
                    np.array([_[1] for _ in batch_x])]


@dataclass
class FragmentGenerator_multi_tax(Sequence):
    def __init__(self, x, y, y_species, weight_classes, classes, seq_len, tax_ranks, max_seq_len=None, k=3, stride=3,
                 batch_size=32, seq_len_like=None, window=False):
        self.x = x
        self.y = y
        self.y_species = y_species
        self.weight_classes = weight_classes
        self.classes = classes
        self.seq_len = seq_len
        self.tax_ranks = tax_ranks
        self.max_seq_len = max_seq_len
        self.k = k
        self.stride = stride
        self.batch_size = batch_size
        self.seq_len_like = seq_len_like
        self.window = window

        from utils.tax_entry import TaxidLineage
        tlineage = TaxidLineage()
        self.tlineage = tlineage
        self.class_vectors = dict()
        for tax_rank in self.classes:
            self.class_vectors.update({tax_rank: get_class_vectors(self.classes[tax_rank])})

        self.token_dict = get_token_dict(ALPHABET, k=3)
        if (self.max_seq_len is None):
            self.max_seq_len = self.seq_len

    def get_class_vectors_multi_tax(self, taxid):
        vector = []
        weight = []

        ranks = self.tlineage.get_ranks(taxid, ranks=self.tax_ranks)

        # calc vector per tax and append
        # if class not in dict use vector of unknown class
        for index, class_tax_i in enumerate(ranks):
            # try:
            vector.append(self.class_vectors[self.tax_ranks[index]].get(ranks[class_tax_i][1],
                                                                    self.class_vectors[self.tax_ranks[index]][
                                                                        'unknown']))
            # except:
            #     print(self.tax_ranks[index])
            #     print(ranks[class_tax_i][1])

        # calc sample weight
        for index, class_tax_i in enumerate(ranks):
            weight.append(self.weight_classes[self.tax_ranks[index]].get(ranks[class_tax_i][1],
                                                                         self.weight_classes[self.tax_ranks[index]][
                                                                             'unknown']))

        return vector, weight

    # def __post_init__(self):
    #     # from utils.tax_entry import TaxDB
    #     # self.taxDB = TaxDB(data_dir="/mnt/fass2/projects/fm_read_classification_comparison/taxonomy")


    def __len__(self):
        return np.ceil(len(self.x)
                       / float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        batch_fragments = self.x[idx * self.batch_size:
                                 (idx + 1) * self.batch_size]
        batch_x = [seq2tokens(seq, self.token_dict, seq_length=self.seq_len,
                              max_length=self.max_seq_len,
                              k=self.k, stride=self.stride, window=self.window,
                              seq_len_like=self.seq_len_like)
                   for seq in batch_fragments]
        if (self.y is not None and len(self.y) != 0):
            batch_classes = self.y_species[idx * self.batch_size:
                                           (idx + 1) * self.batch_size]
            batch_y, weights = zip(*[self.get_class_vectors_multi_tax(taxid) for taxid in batch_classes])
            # batch_y, weights = zip(*[self.get_class_vectors_multi_tax(taxid) for taxid in batch_classes])
            X = [np.array([_[0] for _ in batch_x]), np.array([_[1] for _ in batch_x])]
            y = []
            weights_y = []
            for index in range(len(self.tax_ranks)):
                y.append(np.array([_[index] for _ in batch_y]))
                weights_y.append(np.array([_[index] for _ in weights]))
            return (X, y, weights_y)
        else:
            return [np.array([_[0] for _ in batch_x]),
                    np.array([_[1] for _ in batch_x])]

def get_fine_model(pretrained_model_file):
    # with mirrored_strategy.scope():
    model_fine = generate_bert_with_pretrained(
        pretrained_model_file, len(classes))
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = model_fine.input_shape[0][1]
    return model_fine, max_length


def get_fine_model_multi_tax(pretrained_model_file, num_classes, tax_ranks):
    # with mirrored_strategy.scope():
    model_fine = generate_bert_with_pretrained_multi_tax(pretrained_model_file, num_classes, tax_ranks)
    model_fine.compile(keras.optimizers.Adam(learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    max_length = model_fine.input_shape[0][1]
    # tf.keras.utils.plot_model(model_fine, to_file="model.png", show_shapes=True)
    # model_fine.summary()

    return model_fine, max_length

def prepare_training_val_weights_for_multitax(train_x, train_y, train_y_species,classes_preset=None, unknown_thr=8000, gen_test_set=False):
    """
    1. find classes of interest
    2. split train and val set
    3. identify which of the classes now do not match old threshold
    4. adapt threshold and recalculate classes, weights, etc

    unknown: set to total samples per class - samples in test set
    e.g. total 10000 and 2000 in test set --> 8000
    """
    if classes is None:
        classes_, weight_classes_, species_list_y_ = get_classes_and_weights_multi_tax(train_y_species,
                                                                                    tax_ranks=tax_ranks, unknown_thr=unknown_thr,
                                                                                    norm_weights=norm_weights)
    else:
        classes_ = classes_preset
    train_x = list(zip(train_x, train_y_species))

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.05, stratify=train_y)

    if gen_test_set:
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

    train_x, train_y_species = zip(*train_x)
    val_x, val_y_species = zip(*val_x)
    train_x, val_x = [np.array(i, dtype=object) for i in [train_x, val_x]]
    train_y_species, val_y_species = [np.array(i) for i in [train_y_species, val_y_species]]
    # train_idx = train_y.index.values
    # train_taxID = species_list_y_[train_idx]

    classes_, weight_classes_, species_list_y_ = get_classes_and_weights_multi_tax(train_y_species,
                                                                                   classes_preset=classes_,
                                                                                tax_ranks=tax_ranks,
                                                                                norm_weights=norm_weights)

    if gen_test_set:
        test_x, test_y_species = zip(*test_x)
        return train_x, train_y, train_y_species, val_x, val_y, val_y_species, classes_, weight_classes_, species_list_y_, test_x, test_y, test_y_species

    return train_x, train_y, train_y_species, val_x, val_y, val_y_species, classes_, weight_classes_, species_list_y_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='fine-tune BERT on pre-generated fragments')
    parser.add_argument('pretrained_bert')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--seq_len_like', default=None,
                        help='path of pickled class dict of seq lens for '
                             'generating sampled sequence sizes')
    parser.add_argument('--k', help=' ', default=3, type=int)
    parser.add_argument('--stride', help=' ', default=3, type=int)
    parser.add_argument('--batch_size', help=' ', type=int, default=32)
    parser.add_argument('--epochs', help=' ', type=int, default=4)
    parser.add_argument('--nr_seqs', help=' ', type=int, default=100_000_000)
    parser.add_argument('--learning_rate', help=' ', type=float, default=5e-5)
    parser.add_argument('--save_name',
                        help='custom name for saved finetuned model',
                        default=None)
    parser.add_argument('--store_predictions', help=' ', action='store_true')
    parser.add_argument('--store_train_data', help=' ', action='store_true')
    parser.add_argument('--roc_auc', help=' ', action='store_true')
    parser.add_argument('--multi_tax', help=' ', action='store_true')
    parser.add_argument('--test_benchmark', help=' ', action='store_true')

    args = parser.parse_args()

    # tax_ranks = ["superkingdom", "phylum", "genus"]
    tax_ranks = ["superkingdom", "phylum"]
    test = False
    norm_weights = True

    learning_rate = args.learning_rate
    if (args.seq_len_like is not None):
        seq_len_dict = pickle.load(open(args.seq_len_like, 'rb'))
        min_nr_seqs = min(map(len, seq_len_dict.values()))
        seq_len_like = []
        for k in seq_len_dict:
            seq_len_like.extend(np.random.choice(seq_len_dict[k], min_nr_seqs)
                                // args.k)
    else:
        seq_len_like = None

    # model, max_length = get_fine_model_multi_tax(args.pretrained_bert, num_classes=(2,17,22),tax_ranks=tax_ranks)
    # exit()
    if not args.test_benchmark:
        # loading training data

        if args.multi_tax:
            x, y, y_species = load_fragments(args.fragments_dir, balance=False, nr_seqs=args.nr_seqs)
            f_train_x, f_train_y, f_train_y_species, f_val_x, f_val_y, f_val_y_species, classes, weight_classes, species_list_y,  test_x, test_y, \
            test_y_species = prepare_training_val_weights_for_multitax(x, y, y_species, unknown_thr=8000,
                                                                                 gen_test_set=True)

        else:
            x, y, y_species = load_fragments(args.fragments_dir, nr_seqs=args.nr_seqs)
            f_train_x, f_test_x, f_train_y, f_test_y = train_test_split(
                x, y, test_size=0.2, stratify=y)
            f_train_x, f_val_x, f_train_y, f_val_y = train_test_split(
                f_train_x, f_train_y, test_size=0.05, stratify=f_train_y)

    else:
        if test:
            f_test_x, f_test_y, f_test_y_species = load_dataset(os.path.join(args.fragments_dir,"test.tsv"))
            f_train_x, f_train_y, f_train_y_species = load_dataset(os.path.join(args.fragments_dir,"train.tsv"))
            classes = pickle.load(open(os.path.join(args.fragments_dir,"classes.pkl"),'rb'))

        else:
            f_test_x, f_test_y, f_test_y_species = load_dataset(os.path.join(args.fragments_dir,"test.tsv"))
            f_train_x, f_train_y, f_train_y_species = load_dataset(os.path.join(args.fragments_dir,"train.tsv"))
            classes = pickle.load(open(os.path.join(args.fragments_dir,"classes.pkl"),'rb'))


        f_train_x, f_train_y, f_train_y_species, f_val_x, f_val_y, f_val_y_species, classes, weight_classes, species_list_y = prepare_training_val_weights_for_multitax(f_train_x, f_train_y, f_train_y_species, classes_preset=classes)


    if test:
        from models.bert_utils import load_bert
        # model = load_bert("/home/go96bix/projects/dna_class/resources/bert_nc_C2_filtered_model.best.loss.hdf5", compile_=True)
        model = load_bert("/home/go96bix/projects/dna_class/resources/bert_nc_C2_big_trainingset_all_norm_weights_model.best.loss.hdf5", compile_=True)
        max_length = model.input_shape[0][1]
    else:
        # building model
        if args.multi_tax:
            num_classes = [len(classes[tax].keys()) for tax in classes]
            model, max_length = get_fine_model_multi_tax(args.pretrained_bert, num_classes=num_classes, tax_ranks=tax_ranks)
        else:
            model, max_length = get_fine_model(args.pretrained_bert)

        if (args.seq_len > max_length):
            warning(f'desired seq len ({args.seq_len}) is higher than possible ({max_length})'
                    f'setting seq len to {max_length}')
            args.seq_len = max_length
    generator_args = {
        'max_seq_len': max_length, 'k': args.k, 'stride': args.stride,
        'batch_size': args.batch_size, 'window': True,
        'seq_len_like': seq_len_like}
    model.summary()

    if not test:
        name="_small_trainingset_filtered_fix_classes_selection"
        # name="_all"
        filepath1 = splitext(args.pretrained_bert)[0] +name+ "_model.best.acc.hdf5"
        filepath2 = splitext(args.pretrained_bert)[0] +name+ "_model.best.loss.hdf5"
        checkpoint1 = ModelCheckpoint(filepath1, monitor='val_phylum_out_accuracy', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='max')
        checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='min')
        checkpoint3 = EarlyStopping('val_loss', min_delta=0, patience=2, restore_best_weights=True)
        tensorboard_callback = TensorBoard(log_dir="./logs/run_broad_learning"+name, histogram_freq=1, write_graph=True,
                                           write_images=True, update_freq=100, embeddings_freq=1)

        # callbacks_list = [checkpoint1, checkpoint2, checkpoint3]
        callbacks_list = [checkpoint1, checkpoint2, checkpoint3, tensorboard_callback]

        if (args.store_train_data):
            from datetime import datetime

            time_str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
            for kind, x, y in [('train', f_train_x, f_train_y), ('val', f_val_x, f_val_y),
                               ('test', f_test_x, f_test_y)]:
                with open(f'{time_str}_{kind}_data.json', 'w') as f:
                    json.dump([x, y], f)
            print('saved train/test/val data.')

    if args.multi_tax:
        test_g = FragmentGenerator_multi_tax(f_test_x, f_test_y, f_test_y_species, weight_classes, seq_len=args.seq_len,
                                             tax_ranks=tax_ranks, classes=classes, **generator_args)
    else:
        test_g = FragmentGenerator(f_test_x, f_test_y, args.seq_len, **generator_args)

    if not test:
        try:
            if args.multi_tax:
                model.fit(
                    FragmentGenerator_multi_tax(f_train_x, f_train_y, f_train_y_species, weight_classes, seq_len=args.seq_len,
                                                tax_ranks=tax_ranks, classes=classes, **generator_args),
                    callbacks=callbacks_list, epochs=args.epochs,
                    validation_data=FragmentGenerator_multi_tax(f_val_x, f_val_y, f_val_y_species, weight_classes,
                                                                seq_len=args.seq_len, tax_ranks=tax_ranks, classes=classes,
                                                                **generator_args), verbose=2)
            else:
                model.fit(FragmentGenerator(f_train_x, f_train_y, args.seq_len, **generator_args),
                          callbacks=callbacks_list, epochs=args.epochs,
                          validation_data=FragmentGenerator(f_val_x, f_val_y, args.seq_len, **generator_args))
        except (KeyboardInterrupt):
            print("training interrupted, current status will be saved and tested, press ctrl+c to cancel this")
            file_suffix = '_aborted.hdf5'
            model.save(splitext(args.pretrained_bert)[0] + '_aborted.h5')
            print('testing...')
            result = model.evaluate(test_g)
            print("test results:",*zip(model.metrics_names, result))
            exit()
        if (args.save_name is not None):
            save_path = args.save_name + '.h5'
        else:
            save_path = os.path.splitext(args.pretrained_bert)[0] + '_finetuned.h5'
        model.save(save_path)
        print('testing...')

    if (args.store_predictions or args.roc_auc):
        predicted = predict(
            model, test_g,
            args.roc_auc, classes, return_data=args.store_predictions, calc_metrics=False)
        y_true, y_pred = predicted["data"]
        # !!! only needed for small fragment set !!!
        # val_g = FragmentGenerator_multi_tax(f_val_x, f_val_y, f_val_y_species, weight_classes, seq_len=args.seq_len,
        #                                     tax_ranks=tax_ranks, classes=classes, **generator_args)
        # predicted_val = predict(
        #     model, val_g,
        #     args.roc_auc, classes, return_data=args.store_predictions, calc_metrics=False)
        # y_true_val, y_pred_val = predicted_val["data"]
        #
        for i in range(len(y_pred)):
        #     np_val = pd.crosstab(np.argmax(np.array(y_true_val[i]), axis=1), np.argmax(np.array(y_pred_val[i]), axis=1)).values
        #     # reorder output according to best val prediction
        #     print(np.argmax(np_val, axis=1), f'all {len(np.argmax(np_val, axis=1))}', f'unique {len(np.unique(np.argmax(np_val, axis=1)))}')
        #     unq, unq_idx, unq_cnt = np.unique(np.argmax(np_val, axis=1), return_inverse=True, return_counts=True)
        #     cnt_mask = unq_cnt > 1
        #     print(f'duplicates {unq[cnt_mask]}')
        #     y_pred_i_sorted = y_pred[i][:, np.argmax(np_val, axis=1)]
        #     acc = balanced_accuracy_score(np.argmax(y_true[i],axis=1),np.argmax(y_pred_i_sorted,axis=1))
        #     print(f"{test_g.tax_ranks[i]} acc:", acc)
        #     print(pd.crosstab(np.argmax(np.array(y_true[i]),axis=1), np.argmax(np.array(y_pred_i_sorted),axis=1),rownames=['True'], colnames=['Predicted'], margins=True))
        #     predicted["data"][1][i]=y_pred_i_sorted
        #     # sorted_names = np.array(sorted(classes))
        #     # print(pd.crosstab(sorted_names[np.argmax(np.array(y_true[i]),axis=1)], sorted_names[np.argmax(np.array(y_pred[i]),axis=1)],
        #     #                   rownames=['True'], colnames=['Predicted'], margins=True))
            acc = balanced_accuracy_score(np.argmax(y_true[i], axis=1), np.argmax(y_pred[i], axis=1))
            print(f"{test_g.tax_ranks[i]} acc:", acc)
            print(pd.crosstab(np.argmax(np.array(y_true[i]), axis=1), np.argmax(np.array(y_pred[i]), axis=1),
                              rownames=['True'], colnames=['Predicted'], margins=True))
        result = predicted['metrics']
        metrics_names = predicted['metrics_names']
        if (args.store_predictions):
            import pickle

            if test:
                pickle.dump(predicted, open("/home/go96bix/projects/dna_class/resources/" + "big_trainingset_all_normed"
                                    + '_predictions.pkl', 'wb'))
            else:
                pickle.dump(predicted, open(os.path.splitext(save_path)[0]
                                            + '_predictions.pkl', 'wb'))

    else:
        result = model.evaluate(test_g)
        metrics_names = model.metrics_names
    print("test results:", *zip(metrics_names, result))
