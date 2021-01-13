import torch
import tensorflow as tf
from utils.visualize_bert import keras2torch
from models.bert_utils import load_bert
import numpy as np
from tokenizers import BertWordPieceTokenizer
from models.bert_utils import get_token_dict
from preprocessing.generate_data import seq2tokens, ALPHABET, seq2kmers
from bertviz.bertviz.util import format_attention
from bertviz.bertviz import head_view, model_view
import json
import argparse
from Bio import SeqIO
import webbrowser


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('fasta')
    parser.add_argument('model')
    parser.add_argument('-a', type=int, help='sequence start (default: 0)',
                        default=0)
    parser.add_argument('-n', type=int, help='sequence size (default: 500)',
                        default=500)
    return parser.parse_args()

def js_data(tmodel, in_ids):
    out = tmodel(input_ids=torch.tensor(np.array([in_ids]), dtype=torch.long), output_attentions=True)
    attn = format_attention(out[-1]).tolist()
    tokens = list(map({v: k for k, v in tokend.items()}.__getitem__, in_ids))
    return {'attn': attn, 'left_text': tokens, 'right_text': tokens}

def ins_json(tmodel, in_ids, out_handle):
    out_handle.write('PYTHON_PARAMS = ');
    json.dump({'default_filter': 'all', 'root_div_id': 'bert_viz',
               'attention': {'all': js_data(tmodel, in_ids)}},
              out_handle, indent=2)

if __name__ == '__main__':
    args = parse_arguments()
    seq = str(list(SeqIO.parse(open(args.fasta), 'fasta'))[0].seq)
    tokend = get_token_dict()
    # load & convert model
    m = load_bert(args.model)
    model_len = m.layers[0].input_shape[0][1]
    # converted 🤗 transformers model
    tm = keras2torch(m)
    _ = tm.eval()                   # toggle evaluation mode, no output
    # whole sequence to tokens
    ins = seq2tokens(seq, tokend, model_len)
    # extract specified part
    ins_extract = np.concatenate((ins[0][:1], ins[0][args.a : args.a + args.n - 2],
                                  ins[0][-1:]))
    # dump json
    ins_json(tm, ins_extract, open('view_data.js', 'w'))
    html = """
    <html>
      <body>
        <div id='bert_viz'>
          <span style="user-select:none">
            Layer: <select id="layer"></select>
          </span>
          <div id='vis'></div>
        </div>
        <script src="https://requirejs.org/docs/release/2.3.6/minified/require.js"></script>
        <script>
          require.config({
              paths: {
                  d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',
                  jquery: 'https://ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
              }
          });
        </script>
        <script src="view_data.js"></script>
        <script src="bertviz/bertviz/head_view.js"></script>
      </body>
    </html>"""
    with open('view_model.html', 'w') as f:
        f.write(html)
    webbrowser.open('view_model.html')
