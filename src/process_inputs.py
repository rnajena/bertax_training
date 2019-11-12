"""Library for various sequence preprocessing/encoding functions"""


from Bio import SeqIO
from tqdm import tqdm
from numpy import floor, ceil, where


ALPHABET = 'ACGT'
SPECIAL_INDEX = 64

######
# IO #
######


def read_raw_inputs(fasta_in) -> list:
    with open(fasta_in) as f:
        records = list(SeqIO.parse(f, 'fasta'))
    return [str(record.seq) for record in records]


def read_seq(file_name):
    """reads everything (except header(s)) in fasta file as one sequence"""
    with open(file_name) as f:
        seq = ''.join(line.strip() for line in f
                      if not line.startswith('>'))
    return seq


def read_headers(fasta_in):
    with open(fasta_in) as f:
        headers = [line.strip().lstrip('>') for line in
                   f.readlines() if line.startswith('>')]
    return headers

#####################
# Sequence to words #
#####################


def seq2kmers(seq, k=3, stride=3, pad=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmers.append(seq[i:i+k])
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i+k:].ljust(k, 'N'))
    return kmers

#####################
# Words to encoding #
#####################


def kmers2index(word_seq, handle_nonalph='split'):
    seq_enc = []
    replacements = {'N': 'ACGT',
                    'Y': 'CT',
                    'R': 'AG',
                    'S': 'GC',
                    'W': 'AT',
                    'K': 'GT',
                    'M': 'AC',
                    'B': 'CGT',
                    'D': 'AGT',
                    'H': 'ACT',
                    'V': 'ACG'}
    for word in word_seq:
        if len(word) != 3:
            raise Exception('only 3-mers are supported by this method')
        if all(letter in ALPHABET for letter in word):
            seq_enc.append(kmer2index(word))
        else:
            if (handle_nonalph == 'split'):
                words = [word.replace(amb_l, new_l) for amb_l in replacements
                         for new_l in replacements[amb_l] if amb_l in word]
                seq_enc.append([kmer2index(w) for w in words])
            else:
                seq_enc.append(SPECIAL_INDEX)
    return seq_enc


def kmers2onehot(word_seq, handle_nonalph='split'):
    max_index = 64 if handle_nonalph != 'split' else 63
    return [index2onehot(index, max_index) for index in
            kmers2index(word_seq, handle_nonalph)]


def kmers2base64(word_seq):
    if 'b64' not in globals():
        gen_b64()
    return [b64[index] for index in kmers2index(word_seq,
                                                handle_nonalph='special')]

def

####################
# Word to encoding #
####################


def kmer2index(word):
    """NOTE: only implemented for k==3"""
    return sum([4**(2-pos) * ALPHABET.find(c) for pos, c in
                enumerate(word)])

##################
# Inter encoding #
##################


def b642index(b64char):
    if 'b64' not in globals():
        gen_b64()
    return b64.index(b64char)


def onehot2index(onehot):
    if (1 in onehot):
        return where(onehot == 1)[0][0]
    else:
        return SPECIAL_INDEX


def index2onehot(index_or_indices, max_index=63):
    if (isinstance(index_or_indices, int)):
        indices = [index_or_indices]
    else:
        indices = index_or_indices
    return([(1/len(indices) if i in indices else 0) for i in
            range(max_index+1)])

####################
# Top-level encode #
####################


def encode_inputs(inputs, enc=kmers2onehot, progress=False):
    it = inputs if not progress else tqdm(inputs)
    return [enc(_) for _ in it]


def encode_sequence(seq, max_seq_len, pad=True, method=kmers2onehot,
                    resolution_method=seq2kmers,
                    **kwargs):
    seq = method(resolution_method(seq), **kwargs)
    if (pad):
        seq = pad_sequence(seq, max_seq_len,
                           method='onehot'
                           if method == kmers2onehot else 'index',
                           pos='end', cut=True)
        return seq


def pad_sequence(seq, max_seq_len, method='onehot', pos='balanced', cut=True):
    if (cut and len(seq) > max_seq_len):
        return seq[:max_seq_len]
    if (method == 'onehot'):
        pad_el = [0 for i in range(len(seq[0]))]
    elif (method == 'index'):
        pad_el = SPECIAL_INDEX
    else:
        raise Exception('not implemented yet')
    if (pos == 'balanced'):
        padded = (floor((max_seq_len - len(seq))/2) * [pad_el]
                  + seq + ceil((max_seq_len - len(seq))/2) * [pad_el])
    elif (pos == 'front'):
        padded = (max_seq_len - len(seq)) * [pad_el] + seq
    else:
        padded = seq + (max_seq_len - len(seq)) * [pad_el]
    return padded

##########
# Decode #
##########


def index2kmer(index):
    word = ''
    for i in reversed([4**0, 4**1, 4**2]):
        val = (index // i)
        word += ALPHABET[val]
        index -= i * val
    return word


def encoded2fasta(encoded, headers, fasta_out):
    assert (len(encoded) == len(headers)), (
        'size of headers and encoded sequences doesn\'t match: '
        f'{len(encoded)}!={len(headers)}')
    with open(fasta_out, 'w') as f:
        for header, enc in zip(headers, encoded):
            f.write('>' + header + '\n')
            f.write(''.join(enc) + '\n')


def translate_back(enc_seq, encoding=kmers2base64):
    if (encoding == kmers2base64):
        return [index2kmer(b642index(c)) for c in enc_seq]
    elif (encoding == kmers2onehot):
        return [index2kmer(onehot2index(c)) for c in enc_seq]


#########
# Utils #
#########

def transform_inputs(inputs, progress=False, transform=seq2kmers,
                     *transform_args):
    """transforms raw input into words, returning a list of lists"""
    it = inputs if not progress else tqdm(inputs)
    return [transform(_, *transform_args) for _ in it]


def get_class_vectors(classes) -> dict:
    return {c: [1 if c_i == i else 0 for i in range(len(classes))]
            for c_i, c in enumerate(classes)}


def gen_b64():
    global b64
    ranges = []
    ranges.extend(list(range(ord('A'), ord('Z')+1)))
    ranges.extend(list(range(ord('a'), ord('z')+1)))
    ranges.extend(list(range(ord('0'), ord('9')+1)))
    ranges.extend([ord(_) for _ in ['+', '/', '=']])
    b64 = [chr(_) for _ in ranges]


def process(fasta_in, enc, progress=False) -> list():
    """applies transformation and encoding on input, returning a list of
    sequences. Each sequence is a list of words, with words being ints
    """
    raw_inputs = read_raw_inputs(fasta_in)
    transformed = transform_inputs(raw_inputs, progress)
    encoded = encode_inputs(transformed, enc, progress)
    return encoded


if __name__ == '__main__':
    """When executed as a script, transforms fasta into pseudo-base64-encoded
    sequence of 3-mers(stride=k=3)"""
    from sys import argv, stderr
    if (len(argv) != 3):
        print(f'Usage: python {argv[0]} <fasta_in> <fasta_out>',
              file=stderr)
        exit(-1)
    fasta_in = argv[1]
    fasta_out = argv[2]
    print(f'processing...', file=stderr)
    encoded = process(fasta_in, kmers2base64, progress=True)
    print(f'reading headers...', file=stderr)
    headers = read_headers(fasta_in)
    print(f'writing...', file=stderr)
    encoded2fasta(encoded, headers, fasta_out)
