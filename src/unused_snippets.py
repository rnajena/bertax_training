@dataclass
class DataGenerator:
    root_fa_dir: str
    nr_seqs: int
    classes: list
    enc_dimension: int = 64
    max_seq_len: int = 100

    @DeprecationWarning
    def get_seqs_for_dir(fasta_dir):
        seqs = []
        headers = []
        for fasta in listdir(fasta_dir):
            if not fasta.endswith('.fa'):
                continue
            with open(os.path.join(fasta_dir, fasta)) as f:
                current_seq = ''
                for line in f:
                    line = line.strip()
                    if (len(line) == 0):
                        continue
                    if (line.startswith('>')):
                        # header
                        headers.append(line)
                        if (current_seq != ''):
                            seqs.append(current_seq)
                            current_seq = ''
                    else:
                        current_seq += line
        if (current_seq != ''):
            seqs.append(current_seq)
        assert len(seqs) == len(headers), 'unequal amount of headers/sequences'
        return (headers, seqs)

    @DeprecationWarning
    def get_seq_subset(self, filter_length=None):
        self.seqs = {}
        self.headers = {}
        for c in self.classes:
            class_dir = os.path.join(self.root_fa_dir, c)
            if (not os.path.isdir(class_dir)):
                raise Exception(f'class {c}\'s fasta dir doesn\'t exist',
                                class_dir)
            headers_full, seqs_full = DataGenerator.get_seqs_for_dir(class_dir)
            if (filter_length is not None):
                if hasattr(filter_length, '__iter__'):
                    flen, margin = filter_length
                else:
                    flen, margin = filter_length, 0.05
                indices = [i for i, seq in enumerate(seqs_full)
                           if len(seq) >= flen - margin*flen
                           and len(seq) <= flen + margin*flen]
                headers_full = [headers_full[i] for i in indices]
                seqs_full = [seqs_full[i] for i in indices]
            seqs_c = []
            headers_c = []
            if (self.nr_seqs > len(headers_full)):
                raise Exception(f'not enough sequences for class {c}',
                                len(headers_full), self.nr_seqs)
            for i in sample(range(len(headers_full)), self.nr_seqs):
                seqs_c.append(seqs_full[i])
                headers_c.append(headers_full[i])
            self.seqs[c] = seqs_c
            self.headers[c] = headers_c

    def encode_sequence(seq, max_seq_len, pad=True):
        seq = kmers2onehot(seq2kmers(seq))
        if (pad):
            seq = pad_sequence(seq, max_seq_len, pos='end', cut=True)
        return seq

    def encode_sequences(self, pad=True):
        """encodes sequences inplace

        transforms sequences to sequences of 3-mers and those into one-hot
        vectors."""
        self.max_seq_len = 100
        for i in range(self.nr_seqs):
            for c in self.classes:
                self.seqs[c][i] = kmers2onehot(seq2kmers(self.seqs[c][i]))
                if (len(self.seqs[c][i]) > self.max_seq_len):
                    self.max_seq_len = len(self.seqs[c][i])
        # (optional) padding, now that max seq len is known
        if (pad):
            for i in range(self.nr_seqs):
                for c in self.classes:
                    self.seqs[c][i] = pad_sequence(self.seqs[c][i],
                                                   self.max_seq_len, pos='end')

    def get_class_vectors(self) -> dict:
        return {c: [1 if c_i == i else 0 for i in range(len(self.classes))]
                for c_i, c in enumerate(self.classes)}

    @DeprecationWarning
    def get_data(self):
        # TODO: does it have to be shuffled?
        X = np.empty((self.nr_seqs * len(self.classes), self.max_seq_len,
                      self.enc_dimension))
        for c_i, c in enumerate(self.classes):
            for i in range(self.nr_seqs):
                X[c_i*self.nr_seqs+i] = self.seqs[c][i]

        class_vectors = self.get_class_vectors()
        y = []
        for c in self.classes:
            y.extend([class_vectors[c]] * self.nr_seqs)
        Y = np.array(y)
        return X, Y

    def read_seq(self, file_name):
        """reads everything (except header(s)) in fasta file as one sequence"""
        with open(os.path.join(self.root_fa_dir, file_name)) as f:
            seq = ''.join(line.strip() for line in f
                          if not line.startswith('>'))
        return seq

    def det_max_seq_len(self):
        """determines the maximum length of all the stored fasta files'
        sequences"""
        # NOTE: *3 for 3-mer encoding
        for file_name in tqdm(self.file_names):
            length = 3 * len(self.read_seq(file_name))
            if (length > self.max_seq_len):
                self.max_seq_len = length


# class BatchGenerator(Sequence, DataGenerator):
#     def __init__(self, batch_size, root_fa_dir, nr_seqs, classes,
#                  enc_dimension=64, max_seq_len=1000, train_test_split=0.2,
#                  val_split=0.05,
#                  cache_file=None, force_max_len=True):
#         super(BatchGenerator, self).__init__(
#             root_fa_dir, nr_seqs, classes, enc_dimension, max_seq_len)
#         self.batch_size = batch_size
#         self.train_test_split = train_test_split
#         self.data_mode = 'train'
#         info('read fasta file names')
#         self.get_fa_files(cache_file)
#         info('balance/split dataset')
#         self.process_fa_files()
#         if (not force_max_len):
#             info('determine max seq length')
#             self.det_max_seq_len()
#         self.nr_seqs_train = int(np.ceil((1-self.train_test_split)
#                                          * self.nr_seqs))
#         self.test_batch_offset = np.ceil((self.nr_seqs_train *
#                                           len(self.classes)) /
#                                          float(self.batch_size)).astype(np.int)

#     def __len__(self):
#         nr_seqs = (self.nr_seqs_train if self.data_mode == 'train'
#                    else self.nr_seqs - self.nr_seqs_train)
#         return np.ceil((nr_seqs * len(self.classes)) /
#                        float(self.batch_size)).astype(np.int)

#     def __getitem__(self, idx):
#         idx += 0 if self.data_mode == 'train' else self.test_batch_offset
#         batch_x = self.file_names[idx * self.batch_size:
#                                   (idx+1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:
#                               (idx+1) * self.batch_size]

#         class_vectors = self.get_class_vectors()
#         return (
#             np.array([DataGenerator.encode_sequence(
#                 self.read_seq(file_name), self.max_seq_len, pad=True)
#                       for file_name in batch_x]),
#             np.array([class_vectors[label] for label in batch_y]))

def test_dg():
    # test no batch
    d = DataGenerator('../../sequences/dna_sequences/', 10,
                      ['Viruses', 'Archaea'])
    d.get_seq_subset()
    d.encode_sequences()
    print(d.get_data())
    # test batch
    d = DataGenerator('../../sequences/dna_sequences/', 100,
                      ['Viruses', 'Archaea'])
    d.get_fa_files()
    d.process_fa_files()
    d.det_max_seq_len()
    print(d.max_seq_len)
    print(d.file_names[-5:])
    print(d.labels[-5:])
    print(len(d.file_names), len(d.labels))


def test_bg():
    # test batchGenerator
    b = BatchGenerator(100, '../../sequences/dna_sequences/', 10000,
                       ['Viruses', 'Archaea', 'Bacteria', 'Eukaryota'],
                       max_seq_len=10000,
                       cache_file='../../sequences/dna_sequences/files.json')

# def indices(word_seq, handle_nonalph='ignore'):
#     global enc_dict
#     global enc_ind
#     if ('enc_dict' not in globals()):
#         enc_dict = {}
#     if ('enc_ind' not in globals()):
#         enc_ind = 0
#     enc_seq = []
#     for word in word_seq:
#         if any(letter not in ALPHABET for letter in word):
#             if (handle_nonalph == 'ignore'):
#                 continue
#             elif (handle_nonalph == 'replace'):
#                 # TODO: replace with special enc_ind
#                 pass
#             else:
#                 pass
#         if word not in enc_dict:
#             enc_dict[word] = enc_ind
#             enc_ind += 1
#         ind = enc_dict[word]
#         enc_seq.append(ind)
#     return enc_seq


    # def preprocess_data(self, test_split=0.2, desired_seq_len=None):
    #     info(f'preprocessing data, sampling {self.nr_seqs} sequences '
    #          f'for each of the {len(self.classes)} classes')
    #     self.dg = DataGenerator(self.root_fasta_dir, self.nr_seqs,
    #                             self.classes, self.enc_dimension,
    #                             self.max_seq_len)
    #     self.dg.get_seq_subset(desired_seq_len)
    #     self.dg.encode_sequences(pad=True)
    #     X, y = self.dg.get_data()
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_split)
    #     # y_train = np.transpose(y_train)
    #     # y_test = np.transpose(y_test)
    #     self.data = (X_train, X_test, y_train, y_test)
    #     self.max_seq_len = self.dg.max_seq_len
