from gensim.models import Word2Vec
from preprocessing.process_inputs import seq2kmers
from glob import glob
from tqdm import tqdm


class SentenceCorpus(object):

    def __iter__(self):
        for fasta in tqdm(glob(
                '/home/lo63tor/master/sequences/dna_sequences/*/*.fa')):
            with open(fasta) as f:
                current_seq = ''
                for line in tqdm(f, position=1):
                    line = line.strip()
                    if (len(line) == 0):
                        continue
                    if (line.startswith('>')):
                        if (current_seq != ''):
                            yield seq2kmers(current_seq, k=3, stride=3)
                            current_seq = ''
                    else:
                        current_seq += line
                if (current_seq != ''):
                    yield seq2kmers(current_seq, k=3, stride=3)


word2vec = Word2Vec(SentenceCorpus())
word2vec.save('word2vec_model.w2v')
