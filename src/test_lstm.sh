#!/bin/bash

# gru
nice python test_model.py lstm -v --nr_seqs 10_000 --batch_size 500 --epochs 20 --model_name gru1 --lstm_units 32 --dropout_rate 0.1 --early_stopping_restore_weights False --max_seq_len 500 --cell_type gru --bidirectional False
# lstm
nice python test_model.py lstm -v --nr_seqs 10_000 --batch_size 500 --epochs 20 --model_name lstm1 --lstm_units 32 --dropout_rate 0.1 --early_stopping_restore_weights False --max_seq_len 500 --cell_type lstm --bidirectional False
# bi-gru
nice python test_model.py lstm -v --nr_seqs 10_000 --batch_size 500 --epochs 20 --model_name bigru1 --lstm_units 32 --dropout_rate 0.1 --early_stopping_restore_weights False --max_seq_len 500 --cell_type gru --bidirectional True
# bi-lstm
nice python test_model.py lstm -v --nr_seqs 10_000 --batch_size 500 --epochs 20 --model_name bilstm1 --lstm_units 32 --dropout_rate 0.1 --early_stopping_restore_weights False --max_seq_len 500 --cell_type lstm --bidirectional True
