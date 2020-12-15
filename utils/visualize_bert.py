import transformers
import tensorflow as tf
import torch
import numpy as np
from models.bert_utils import get_token_dict
from preprocessing.generate_data import seq2tokens, ALPHABET, seq2kmers
from tokenizers import BertWordPieceTokenizer
from models.bert_utils import load_bert
from transformers import BertTokenizer, TFBertModel, BertModel, BertConfig

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = TFBertModel.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")


embed_dim = 250
seq_len = 502
transformer_num = 12
head_num = 5
feed_forward_dim = 1024
dropout_rate = 0.05

# t = BertWordPieceTokenizer('vocab.txt')
# t.encode(' '.join(seq2kmers('ATAAAAACATTTAANAAANNN'))).tokens

# bc = BertConfig(vocab_size=69, hidden_size=embed_dim,
#                 num_attention_heads=head_num, num_hidden_layers=transformer_num,
#                 intermediate_size=feed_forward_dim, hidden_dropout_prob=dropout_rate,
#                 max_position_embeddings=seq_len)
# bert = BertModel(config=bc)


def keras2torch(kmodel,
                params={'embed_dim': 250, 'seq_len': 502, 'transformer_num': 12,
                        'head_num': 5, 'feed_forward_dim': 1024,
                        'dropout_rate': 0.05, 'vocab_size': 69}):
    tmodel = BertModel(BertConfig(vocab_size=params['vocab_size'],
                                  hidden_size=params['embed_dim'],
                                  num_attention_heads=params['head_num'],
                                  num_hidden_layers=params['transformer_num'],
                                  intermediate_size=params['feed_forward_dim'],
                                  hidden_dropout_prob=params['dropout_rate'],
                                  attention_probs_dropout_prob=params['dropout_rate'],
                                  max_position_embeddings=params['seq_len'],
                                  layer_norm_eps=tf.keras.backend.epsilon() * tf.keras.backend.epsilon()))
    # set torch model tensors to the ones from the keras model
    td = {t[0]: t[1] for t in tmodel.named_parameters()}
    kd = {t.name: t for t in kmodel.weights}
    def set_tensor(tname, karray):
        assert (tshape:=td[tname].detach().numpy().shape) == (
            kshape:=karray.shape), f'{tname} has incompatible shape: {tshape} != {kshape}'
        with torch.no_grad():
            td[tname].data = torch.nn.Parameter(torch.Tensor(karray))
    # 1 INPUT
    t_pfix = 'embeddings.'
    k_pfix = 'Embedding-'
    # set_tensor(t_pfix + 'position_ids', td[t_pfix + 'position_ids']) # don't change
    set_tensor(t_pfix + 'word_embeddings.weight', kd[k_pfix + 'Token/embeddings:0'].numpy())
    set_tensor(t_pfix + 'position_embeddings.weight', kd[k_pfix + 'Position/embeddings:0'].numpy())
    set_tensor(t_pfix + 'token_type_embeddings.weight', kd[k_pfix + 'Segment/embeddings:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.weight', kd[k_pfix + 'Norm/gamma:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.bias', kd[k_pfix + 'Norm/beta:0'].numpy())
    # 2 LAYERS
    for i in range(params['transformer_num']):
        t_pfix_l = f'encoder.layer.{i}.'
        k_pfix_l = f'Encoder-{i+1}-'
        # SELF-ATTENTION
        # NOTE: (embed_dim x embed_dim) matrices have to be transposed!
        t_pfix = t_pfix_l + 'attention.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention/Encoder-{i+1}-MultiHeadSelfAttention_'
        set_tensor(t_pfix + 'self.query.weight', kd[k_pfix + 'Wq:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.query.bias', kd[k_pfix + 'bq:0'].numpy())
        set_tensor(t_pfix + 'self.key.weight', kd[k_pfix + 'Wk:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.key.bias', kd[k_pfix + 'bk:0'].numpy())
        set_tensor(t_pfix + 'self.value.weight', kd[k_pfix + 'Wv:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.value.bias', kd[k_pfix + 'bv:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight', kd[k_pfix + 'Wo:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + 'bo:0'].numpy())
        # NORM
        t_pfix = t_pfix_l + 'attention.output.LayerNorm.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention-Norm/'
        set_tensor(t_pfix + 'weight', kd[k_pfix + 'gamma:0'].numpy())
        set_tensor(t_pfix + 'bias', kd[k_pfix + 'beta:0'].numpy())
        # FF
        t_pfix = t_pfix_l + ''
        k_pfix = k_pfix_l + 'FeedForward'
        set_tensor(t_pfix + 'intermediate.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W1:0'].numpy().transpose())
        set_tensor(t_pfix + 'intermediate.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b1:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W2:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b2:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.weight', kd[k_pfix + '-Norm/gamma:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.bias', kd[k_pfix + '-Norm/beta:0'].numpy())
    # 3 OUTPUT (before class)
    set_tensor('pooler.dense.weight', kd['NSP-Dense/kernel:0'].numpy().transpose())
    set_tensor('pooler.dense.bias', kd['NSP-Dense/bias:0'].numpy())
    return tmodel

if __name__ == '__main__':
    test_string = ("CTGACGCACCCGGGTGCCATTCTTGCAAAAAGCCTTACAATTCCGCTGATTGATGCTACGCATTATGCAACTGAACTTCC"
                   "GGGACTGTATCGTTTGCGAGATTTAATCGCTTCCTTTGGAGTCGAGTCAGCGGTATTTGATACTTCTGTTCCATGGAGAA"
                   "TGAAAACCTATTATGAAAATTACTGAGTTAGAACAAAAAAAAGTACCGCACGGTGAAGTTGTCCTCATTGGTCTTGGCCG"
                   "TCTTGGTCTGAGAACAGCCCTAAATCTCATGCATGTCAATCGGGGCGGACCAGTTCGGATAACTGTGTATGACGGACAAA"
                   "AAATATCTGCCGATGATCTGATATTCCGCATGTGGGGTGGAGAAATTGGCGAATATAAAACAGATTTCCTCAAACGGCTT"
                   "GCAGGCCCCGGATACAGCAGGGAAATAATATCAGTTCCAGAGTATATTTCTGAGGAAAATCTGTCTCTGATTACCGGAGG"
                   "GGATGTTGCGTGTGTCGAGATTGCAGGCGGTGATACATTGCCTACTACCGGGGCTATTATCCGGCATGCCCAGTCTTTGG"
                   "GCATGAAGACTATCAGTACGATGGGTGTATTTGGTATTTCCGGCGATAATGTTTATGCCGTTCCTCTGGAAGAAGCAAAT"
                   "ACAGATAATCCAATTGTTGCCGCAATGCTTGAATACGGGATTTCCCATCATATGCTTGTCGGGACTGGAAAACTGATTCG"
                   "TGACTGGGAACCTGTTACTCCGTATATCATGGATAAAATTGCAGAAGTGATGTCGTCAGAAATACTGCGTCTGACCGAGG"
                   "GGAAATAATGCCGACGATATCGACTGCCGAATGCTTTACCCACGGAAAAGTTGCAAATGAGCTCCATGCATTTGCCCGCG"
                   "GGTATCCGCATGAATATCTCTTTTCTATAGATAGGAAAAAAGTTGATATTTCCGTTGTGGCCGGGATGTTTATTCCAACA"
                   "CTTACAGGTGTCAGAACTCTTCTGCATTTTGAGCCGCTGGAACCGCGGTTGGTTATAGACACGGTGAAAGTTTATGAACA"
                   "GGATCAGGATTGTATTATGGCATGCCGGATGGCGGAGGCCGTTATGCGGGTGACCGGGGCAGATATTGGTATAGGAACTA"
                   "CTGCAGGCATCGGGAAAGGCGCAGTGGCAATAGCCTCTCAGGATAAAATCTATTCCAAAGTCACAAGAATTGATGCAGAT"
                   "TTCAGGACTTCAGATGCAAAAAAACTGATGCAGCGTGAAAAGTCAGGTGTTTTTACTGCACTGCGTTTGTTTGAGGAATT"
                   "TTTGTTGGAGGGGGAGTTCCCCGATAGTTATAATAAATACATATAATTAGTAACACAAATTGCTATTAATATTAATATTA"
                   "TAACTACATTAATCATATTGATTTTAACATATTTAGAAAGATTTATTACGAATATTATTAAATACACTATTGTTGTCACA"
                   "TATTGATGGCAGTACAAACTGGAGATTACATACATGAAAGTAGCAATTTTAGGAGCAGGA")

    tokend = get_token_dict()
    t = BertWordPieceTokenizer('resources/vocab.txt')
    ins = seq2tokens(test_string, tokend, 502)
    print(tk:=ins[0])
    print(tk_man:=np.array(t.encode(' '.join(seq2kmers(test_string))).ids))
    print(tk == tk_man)
    # NOTE: tokenizer has the exact same output as the manual version, except for the last token (shouldn't matter)
    

    m = load_bert('resources/bert_nc_C2_final.h5')
    kmodel_out_class = tf.keras.backend.function(m.input, m.get_layer(name='dense_1').output)(
        [np.array([ins[0]]), np.array([ins[1]])])
    assert kmodel_out_class[0, 1] > 0.9 # models predicts Archaea sequence correctly
    kmodel_out = tf.keras.backend.function(m.input, m.get_layer(name='NSP-Dense').output)(
        [np.array([ins[0]]), np.array([ins[1]])])
    tf.keras.Model(inputs=m.input, outputs=[l.output for l in m.layers])(
        [np.array([ins[0]]), np.array([ins[1]])])
    print(tf.keras.Model(inputs=m.input,
                   outputs=[m.get_layer(name='Encoder-1-MultiHeadSelfAttention-Norm').output])(
                       [np.array([ins[0]]), np.array([ins[1]])]))
    print(tf.keras.Model(inputs=m.input,
                   outputs=[m.get_layer(name='Encoder-1-FeedForward-Norm').output])(
                       [np.array([ins[0]]), np.array([ins[1]])]))
    print(tf.keras.Model(inputs=m.input,
                   outputs=[m.get_layer(name='Encoder-1-FeedForward').output])(
                       [np.array([ins[0]]), np.array([ins[1]])]))
    print(tf.keras.Model(inputs=m.input,
                   outputs=[m.get_layer(name='Embedding-Norm').output])(
                       [np.array([ins[0]]), np.array([ins[1]])]))
    tm = keras2torch(m)
    tm.eval()

    

    def kout(m, out, ins):
        return tf.keras.Model(inputs=m.input, outputs=(
            [m.get_layer(name=out).output] if out != 'all' else [l.output for l in m.layers]))(
                [np.array([ins[0]]), np.array([ins[1]])])

    def close_perc(k, t, tol=1e-4):
        return (~np.isclose(k, t.detach().numpy(), atol=tol)).sum() / np.product(k.shape)
    
    # step by step

    # embeddings
    t_embs = tm.embeddings.forward(input_ids=torch.tensor(np.array([ins[0]]), dtype=torch.long))
    k_embs = kout(m, 'Embedding-Norm', ins)
    print((~np.isclose(k_embs, t_embs.detach().numpy(), atol=1e-7)).sum() / np.product(k_embs.shape))

    # ➡️ layer 1 self-attention
    t_att_self = tm.encoder.layer[0].attention.self(t_embs)
    t_att_out = tm.encoder.layer[0].attention.output(t_att_self[0], t_embs)
    k_att = kout(m, 'Encoder-1-MultiHeadSelfAttention-Norm', ins)
    print(close_perc(k_att, t_att_out, 1e-2))
    # NOTE: seems to work till here, although definitely precision lost    

    # ➡️ layer 1 feed-forward
    t_int = tm.encoder.layer[0].intermediate.forward(t_att_out)
    t_int_od = tm.encoder.layer[0].output.dense.forward(t_int)
    
    k_ff = kout(m, 'Encoder-1-FeedForward', ins)
    print(close_perc(k_ff, t_int_od, 1e-2))
    # NOTE: works till here, no further precision lost

    t_int2 = tm.encoder.layer[0].intermediate.forward(torch.tensor(k_att.numpy()))
    t_int_od2 = tm.encoder.layer[0].output.dense.forward(t_int2)
    print(close_perc(k_ff, t_int_od2, 1e-2))
    # NOTE: in fact, most if not all precision lost during self-attention step

    # ➡️ layer 1 feed-forward norm
    t_ff_n = tm.encoder.layer[0].output.LayerNorm.forward(t_int_od)
    print(t_ff_n)
    
    print(kout(m, 'Encoder-1-FeedForward', ins))
    print(kout(m, 'Encoder-1-FeedForward-Dropout', ins))
    print(kout(m, 'Encoder-1-FeedForward-Add', ins))
    print(kout(m, 'Encoder-1-FeedForward-Norm', ins))

    # ➡️ whole layer 1
    t_l1 = tm.encoder.layer[0](t_embs)
    t_l1_2 = tm.encoder.layer[0](torch.tensor(k_embs.numpy()))
    k_l1 = kout(m, 'Encoder-1-FeedForward-Norm', ins)
    print(close_perc(k_l1, t_l1[0], 1e-3))
    print(close_perc(k_l1, t_l1_2[0], 1e-3))
    # NOTE: kinda works

    # ➡️ layer 2
    t_l2 = tm.encoder.layer[1](t_l1[0])
    k_l2 = kout(m, 'Encoder-2-FeedForward-Norm', ins)
    print(close_perc(k_l2, t_l2[0], 1e-2))

    # ➡️ layer 2
    t_l3 = tm.encoder.layer[2](t_l2[0])
    k_l3 = kout(m, 'Encoder-3-FeedForward-Norm', ins)
    print(close_perc(k_l3, t_l3[0], 1e-2))

    # ➡️ all layers
    t_prev = t_embs
    for i in range(12):
        t_l = tm.encoder.layer[i](t_prev)
        k_l = kout(m, f'Encoder-{i+1}-FeedForward-Norm', ins)
        print(f'layer {i+1}', close_perc(k_l, t_l[0], 1e-2))
        t_prev = t_l[0]
    # NOTE: last layer even has one of the lowest values -> works kinda

    # NSP Dense
    t_nsp = tm.pooler(t_prev)
    k_nsp = kout(m, f'NSP-Dense', ins)
    print(close_perc(k_nsp, t_nsp, 1e-2))
    # whole model: kinda works!!
    

    def print_interm(self, input, output):
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        # print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        # print('input size:', input[0].size())
        # print('output size:', output.data.size())
        # print('output norm:', output.data.norm())

    tm.register_forward_hook(print_interm)

    

    
    tmodel_out = tm(input_ids=torch.tensor(np.array([ins[0]]), dtype=torch.long),
                    # token_type_ids=torch.tensor(np.array([ins[1]]), dtype=torch.long),
                    output_hidden_states=True,
                    return_dict=True,
                    output_attentions=True, encoder_hidden_states=True
                    )
