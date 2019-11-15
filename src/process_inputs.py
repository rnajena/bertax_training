"""Library for various sequence preprocessing/encoding functions"""


from Bio import SeqIO
from tqdm import tqdm
from numpy import floor, ceil, where, array


ALPHABET = 'ACGT'

##############
# IO + Utils #
##############


def get_special_index(k):
    return len(ALPHABET)**k


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


def word_type(word_seq):
    if (len(word_seq[0]) == 1):
        # NOTE: for performance reasons
        # NOTE: not entirely safe: assumes length of all words from the first
        # NOTE: len==1 -> nucleotides
        return 'nucleotide'
    else:
        # NOTE: also unsafe: assumes equal length and length==3
        return '3mer'

#####################
# Sequence to words #
#####################


def seq2kmers(seq, k=3, stride=3, pad=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (k == 1 and stride == 1):
        # for performance reasons
        return seq
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmers.append(seq[i:i+k])
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i+k:].ljust(k, 'N'))
    return kmers


def seq2nucleotides(seq):
    """dummy method, just returns seq as is"""
    return seq

#####################
# Words to encoding #
#####################


def words2index(word_seq, handle_nonalph='split'):
    """
    >>> words2index(['ATG', 'GGA', 'TTA'], handle_nonalph='special')
    [14, 40, 60]
    >>> words2index(['ATG', 'GGN', 'TTA'], handle_nonalph='special')
    [14, 64, 60]
    >>> words2index(['ATG', 'GGN', 'TTA'], handle_nonalph='split')
    [14, [40, 41, 42, 43], 60]
    >>> words2index('ATGGGANA', handle_nonalph='special')
    [0, 3, 2, 2, 2, 0, 4, 0]
    >>> words2index('ATGGGANA', handle_nonalph='split')
    [0, 3, 2, 2, 2, 0, [0, 1, 2, 3], 0]
    """
    k = len(word_seq[0])
    if (k == 1 and handle_nonalph == 'special'):
        return [(ALPHABET.find(n)
                 if n in ALPHABET else get_special_index(1))
                for n in word_seq]
    # everything below for k-mers (k==3) (NOTE: unsafe)
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
        if all(letter in ALPHABET for letter in word):
            seq_enc.append(kmer2index(word))
        else:
            if (handle_nonalph == 'split'):
                words = [word.replace(amb_l, new_l) for amb_l in replacements
                         for new_l in replacements[amb_l] if amb_l in word]
                seq_enc.append([kmer2index(w) for w in words])
            else:
                seq_enc.append(get_special_index(k))
    return seq_enc


def words2onehot(word_seq, handle_nonalph='split'):
    """
    >>> ex1 = words2onehot(['ATG', 'GGA', 'TTA'], handle_nonalph='special')[0]
    >>> len(ex1)
    65
    >>> ex1[14]
    1.0
    >>> words2onehot(['ATG', 'GGN', 'TTA'], handle_nonalph='special')[1][-1]
    1.0
    >>> words2onehot(['ATG', 'GGN', 'TTA'], handle_nonalph='split')[1][40:44]
    [0.25, 0.25, 0.25, 0.25]
    >>> words2onehot('ATGGGANA', handle_nonalph='special')[-2:]
    [[0, 0, 0, 0, 1.0], [1.0, 0, 0, 0, 0]]
    """
    k = len(word_seq[0])
    special_index = get_special_index(k)
    max_index = (special_index if handle_nonalph != 'split'
                 else special_index - 1)
    return [index2onehot(index, max_index) for index in
            words2index(word_seq, handle_nonalph)]


def words2base64(word_seq):
    if 'b64' not in globals():
        gen_b64()
    return [b64[index] for index in words2index(word_seq,
                                                handle_nonalph='special')]


####################
# Word to encoding #
####################


def kmer2index(word):
    k = len(word)
    return sum([4**(k - 1 - pos) * ALPHABET.find(c) for pos, c in
                enumerate(word)])

##################
# Inter encoding #
##################


def b642index(b64char):
    if 'b64' not in globals():
        gen_b64()
    return b64.index(b64char)


def onehot2index(onehot, handle_nonalph='special'):
    if (1 in onehot):
        return where(array(onehot) == 1)[0][0]
    else:
        special_index = (len(onehot) if handle_nonalph != 'split'
                         else len(onehot) + 1)
        return special_index


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


def encode_inputs(inputs, enc=words2onehot, progress=False):
    it = inputs if not progress else tqdm(inputs)
    return [enc(_) for _ in it]


def encode_sequence(seq, max_seq_len, pad=True, method=words2onehot,
                    k=3, stride=3,
                    **kwargs):
    """
    >>> encode_sequence('ATGGGG', 3, method=words2index, k=3, stride=3)
    [14, 42, 64]
    >>> encode_sequence('ATGGGG', 5, method=words2index, k=3, stride=1)
    [14, 58, 42, 42, 64]
    >>> encode_sequence('ATGGGG', 5, method=words2index, k=1, stride=1)
    [0, 3, 2, 2, 2]
    """
    seq = method(seq2kmers(seq, k, stride), **kwargs)
    if (pad):
        method_str = 'onehot' if method == words2onehot else 'index'
        seq = pad_sequence(seq, max_seq_len, method=method_str, k=k,
                           pos='end', cut=True)
        return seq


def pad_sequence(seq, max_seq_len, method='onehot',
                 k=3, pos='end', cut=True):
    if (cut and len(seq) > max_seq_len):
        return seq[:max_seq_len]
    if (method == 'onehot'):
        pad_el = [0 for i in range(len(seq[0]))]
    elif (method == 'index'):
        pad_el = get_special_index(k)
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


# TODO: might not work anymore, fix if needed
def translate_back(enc_seq, encoding=words2base64):
    if (encoding == words2base64):
        return [index2kmer(b642index(c)) for c in enc_seq]
    elif (encoding == words2onehot):
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
    encoded = process(fasta_in, words2base64, progress=True)
    print(f'reading headers...', file=stderr)
    headers = read_headers(fasta_in)
    print(f'writing...', file=stderr)
    encoded2fasta(encoded, headers, fasta_out)
