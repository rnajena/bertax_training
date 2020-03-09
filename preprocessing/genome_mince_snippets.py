from Bio import SeqIO
from random import randint, choice
from preprocessing.process_inputs import seq2kmers
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
from pprint import pprint



vgenomes = list(SeqIO.parse(
    open('/mnt/fass1/genomes/Viruses/ncbi_viruses/20190724/all_viruses_blastdb/all_viruses_db.fa'),
    'fasta'))

vgenomes = SeqIO.to_dict(vgenomes, lambda x: x.id.split('|')[1])


def get_fragment(seq, fragment_size=500, k=3):
    """returns a randomly chosen fragment of sequence with given size

    :param SeqRecord seq: SeqRecord from which to extract fragment
    :param fragment_size: size in kmers of the fragment
    :param k: number of nucleotides to group to one kmer
    :returns: fragment of given size or None if fragment_size is greater
              than sequence size
    """
    if (len(seq) // k < fragment_size):
        return None
    start = randint(0, len(seq) - (fragment_size * k))
    end = start + fragment_size * k
    return seq[start:end]
    return seq2kmers(str(seq[start:end].seq))


def kmer_profile(seq, k=7):
    profile = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        profile[kmer] += 1
    return profile


@DeprecationWarning
def pick_fragments(records, nr_fragments, max_its=1_000_000, dist_thr=50):
    profiles = []
    its = 0
    for i in range(nr_fragments):
        fragment = None
        while fragment is None:
            its += 1
            if (its > max_its):
                raise Exception('Maximum iterations reached')
            record = choice(records)
            fragment = get_fragment(record)
            if fragment is None:
                continue
            profile = kmer_profile(str(fragment.seq))
            profiles.append(profile)
            # if (any(kmer_dist(profile, other_profile) < dist_thr
            #         for other_profile in profiles)):
            #     continue
    print('iterations:', its)
    return profiles


def pick_fragment(taxids, profiles, genome_db, max_its=100, dist_thr=50,
                  **get_kwargs):
    """returns one randomly chosen fragment from the SeqRecords of the given taxids

    :param taxids: ids from which to choose, must have entries in genome_db
    :param profiles: kmer-profiles to which to compare the candidate fragment's
                     own profile
    :param genome_db: database allowing access to the SeqRecord given a taxid
    :param max_its: maximum number of attempts until giving up, can be None
    :param dist_thr: minimum score a candidate fragment has to achieve
    :param get_kwargs: kwargs to pass to get_fragment
    :returns: fragment and its kmer-profile; or None if none could be picked
    """
    fragment = None
    its = 0
    while fragment is None:
        its += 1
        if (max_its is not None and its > max_its):
            return None
        record = genome_db[choice(taxids)]
        fragment = get_fragment(record, **get_kwargs)
        if fragment is None:
            continue
        profile = kmer_profile(str(fragment.seq))
        if (any(kmer_dist(profile, other_profile) < dist_thr
                for other_profile in profiles)):
            continue
    return fragment, profile


def kmer_dist(kmers1, kmers2):
    sum_ = 0
    for kmer in set(kmers1).union(set(kmers2)):
        sum_ += (kmers1[kmer] - kmers2[kmer])**2
    return np.sqrt(sum_)


def kmer_dist_np(kmers1, kmers2):
    kmers = list(set(kmers1).union(set(kmers2)))
    return np.linalg.norm(np.array([kmers1[kmer] for kmer in kmers])
                          - np.array([kmers2[kmer] for kmer in kmers]))


def get_sk_fragments(nr_fragments, orders_dict, genome_db,
                     max_its=None, order_max_its=None, dist_thr=50,
                     **get_fragment_kwargs):
    """obtains specified number of fragments based on the given order taxids

    :param nr_fragments: number of fragments to obtain
    :param orders_dict: dictionary, mapping order taxids to species taxids
    :param genome_db: genomeDB containing genomes of the species taxids
    :param max_its: maximum number of iterations until giving up
    :param order_max_its: maximum number of iterations per order
    :param dist_thr: minimum score a candidate fragment has to achieve
    :returns: fragments and kmer-profiles
    """
    def max_its_reached():
        if (max_its is not None):
            return iterations >= max_its
        else:
            return False
    profiles = []
    fragments = []
    iterations = 0
    pbar = tqdm(total=nr_fragments)
    while (len(fragments) < nr_fragments and not max_its_reached()):
        for order in orders_dict:
            iterations += 1
            avail_taxids = [taxid for taxid in orders_dict[order]
                            if taxid in genome_db]
            if (len(avail_taxids)) == 0:
                continue
            picked = pick_fragment(avail_taxids, profiles, genome_db,
                                   order_max_its, dist_thr,
                                   **get_fragment_kwargs)
            if (picked is None):
                continue
            fragment, profile = picked
            fragments.append(fragment)
            profiles.append(profile)
            pbar.update()
    pbar.close()
    return fragments, profiles


@DeprecationWarning
def get_genome(taxid, sk):
    global genome_dbs
    if (taxid in genome_dbs[sk]):
        return genome_dbs[sk]
    else:
        return None


@DeprecationWarning
def get_viral_genome(taxid, mapping, vgenomes):
    if (taxid in mapping):
        return vgenomes[mapping[taxid]]
    else:
        return None


def all_sk_species_ids(sk_order_dict, sk):
    return [sp_taxid for order in sk_order_dict[sk]
            for sp_taxid in sk_order_dict[sk][order]]


def get_virus_mapping(species_ids, mapping_file):
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.strip().split()
            if (len(line) < 2):
                continue
            if (int(line[1]) in species_ids):
                mapping[int(line[1])] = line[0]
    return mapping


sk_order_dict = json.load(
    open('/home/lo63tor/Downloads/new_taxdump/sk_order_dict.json'))

vmapping = get_virus_mapping(
    all_sk_species_ids(sk_order_dict, 'Viruses'),
    'virus_mapping.txt')


fragments, profiles = main_loop(sk_order_dict['Viruses'], 100, sk_order_dict)

seq = choice(vgenomes)
x = get_fragment(seq)
len(x)
prof_x = kmer_profile(str(x.seq))
seq = choice(vgenomes)
y = get_fragment(seq)
len(y)
prof_y = kmer_profile(str(y.seq))

kmer_dist(prof_x, prof_y)

profiles = pick_fragments(vgenomes, 1000)
from itertools import combinations
dists = [kmer_dist(a, b) for a, b in combinations(profiles, 2)]

all_dists = []
def dist_combs(combs):
    all_dists.extend([kmer_dist(x, y) for x, y in combs])

from threading import Thread

combs = list(combinations(profiles, 2))
threads = []
chunk_size = np.ceil(len(combs) / 12).astype(int)
for i in range(12):
    tcombs = combs[i*chunk_size:i*chunk_size+chunk_size]
    t = Thread(target=dist_combs, args=(tcombs,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

import pickle
pickle.dump(all_dists, open('dists.pkl', 'wb'))

all_dists = pickle.load(open('dists.pkl', 'rb'))

import matplotlib.pyplot as plt
plt.hist(all_dists, bins=1000)

import pandas as pd
df = pd.DataFrame({'distances': all_dists})
fig = px.histogram(df, x='distances')
fig.show(renderer='browser')


with open('virus_gis.txt', 'w') as f:
    f.writelines([key + '\n' for key in vgenomes.keys()])
