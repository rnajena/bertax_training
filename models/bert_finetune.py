import keras
from preprocessing.process_inputs import ALPHABET
from preprocessing.generate_data import DataSplit
from models.model import PARAMS
from models.bert_utils import get_token_dict, generate_bert_with_pretrained
from models.bert_utils import seq2tokens, process_bert_tokens_batch
from models.bert_utils import load_bert
import argparse
from os.path import splitext
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')
    parser.add_argument('pretrained_path', help='pretrained model to finetune')
    parser.add_argument('--finetuned', help='provided model is already adapted'
                        ' to being finetuned', action='store_true')
    parser.add_argument('--nr_seqs', default=250_000, type=int,
                        help='nr of sequences to use per class')
    parser.add_argument('--epochs', default=3, type=int,
                        help='BERT paper recommendations: 2, 3, 4')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='BERT paper recommendations: 16, 32')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='BERT paper recommendations: 5e-5, 3e-5, 2e-5')
    parser.add_argument('--root_fa_dir', help=' ',
                        default=PARAMS['data']['root_fa_dir'][1])
    parser.add_argument('--from_cache', help=' ',
                        default=PARAMS['data']['file_names_cache'][1])
    parser.add_argument('--no_balance', help=' ', action='store_true')
    parser.add_argument('--repeated_undersampling', help=' ', action='store_true')
    parser.add_argument('--val_split', help=' ', default=0.05, type=float)
    parser.add_argument('--test_split', help=' ', default=0.2, type=float)
    parser.add_argument('--classes', help=' ', default=PARAMS['data']['classes'][1])
    parser.add_argument('--alphabet', help=' ', default=ALPHABET)
    parser.add_argument('--k', help=' ', default=3, type=int)
    parser.add_argument('--stride', help=' ', default=3, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if (not args.finetuned):
        model_fine = generate_bert_with_pretrained(
            args.pretrained_path, len(args.classes))
    else:
        model_fine = load_bert(args.pretrained_path)
    max_length = model_fine.input_shape[0][1]
    model_fine.summary()
    model_fine.compile(keras.optimizers.Adam(args.learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    token_dict = get_token_dict(args.alphabet, args.k)
    # DataGenerator
    split = DataSplit(root_fa_dir=args.root_fa_dir, nr_seqs=args.nr_seqs,
                      classes=args.classes, from_cache=args.from_cache,
                      train_test_split=args.test_split,
                      val_split=args.val_split, balance=not args.no_balance,
                      repeated_undersampling=args.repeated_undersampling)

    def custom_encode_sequence(seq):
        return seq2tokens(seq, token_dict, seq_length=max_length, window=True,
                          k=args.k, stride=args.stride)
    train_g, val_g, test_g = split.to_generators(
        batch_size=args.batch_size,
        custom_encode_sequence=custom_encode_sequence,
        process_batch_function=process_bert_tokens_batch,
        enc_k=args.k, enc_stride=args.stride)

    # for long run
    filepath1 = splitext(args.pretrained_path)[0] + "model.best.acc.hdf5"
    filepath2 = splitext(args.pretrained_path)[0] + "model.best.loss.hdf5"
    checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='max')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='min')
    checkpoint3 = EarlyStopping('val_accuracy', min_delta=0, patience=args.epochs // 2, restore_best_weights=True)
    callbacks_list = [checkpoint1, checkpoint2, checkpoint3]

    try:
        model_fine.fit(train_g, validation_data=val_g,callbacks=callbacks_list,verbose=1,
                       epochs=args.epochs)
    except (KeyboardInterrupt):
        print("training interrupted, current status will be saved and tested, press ctrl+c to cancel this")
        file_suffix = '_aborted.hdf5'
        model_fine.save(splitext(args.pretrained_path)[0] + file_suffix)
        print('testing...')
        result = model_fine.evaluate(test_g)
        print("test results:",*zip(model_fine.metrics_names, result))
        exit()

    file_suffix = '_finetuned.5' if (not args.finetuned) else '_plus.h5'
    model_fine.save(splitext(args.pretrained_path)[0] + file_suffix)
    print('testing...')
    result = model_fine.evaluate(test_g)
    print("test results:", *zip(model_fine.metrics_names, result))
