from gensim.models import Word2Vec
from process_inputs import seq2kmers

all_words = []
with open('/home/lo63tor/master/sequences/dna_sequences/Viruses/Viruses.fa') as f:
    for line in f:
        line = line.strip()
        if (line.startswith('>') or len(line) == 0):
            continue
        all_words.extend(seq2kmers(line, k=3, stride=3))

word2vec = Word2Vec(all_words)
