if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from random import randint, choice, sample
from preprocessing.process_inputs import seq2kmers, ALPHABET
from preprocessing.genome_db import GenomeDB
from collections import defaultdict
from datasketch import MinHash
import numpy as np
import json
import logging
from tqdm import tqdm
from pprint import pprint
import os.path
from threading import Thread
import argparse

ALPHABET_all = ALPHABET.lower() + ALPHABET.upper()

def kmer_profile(seq, k=7):
    profile = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        profile[kmer] += 1
    return profile


def kmer_dist(kmers1, kmers2):
    sum_ = 0
    for kmer in set(kmers1).union(set(kmers2)):
        sum_ += (kmers1[kmer] - kmers2[kmer])**2
    return np.sqrt(sum_)


def kmer_dist_np(kmers1, kmers2):
    kmers = list(set(kmers1).union(set(kmers2)))
    return np.linalg.norm(np.array([kmers1[kmer] for kmer in kmers])
                          - np.array([kmers2[kmer] for kmer in kmers]))


def minhash(seq, k, s):
    m = MinHash(num_perm=s)
    for word in seq2kmers(seq, k=k, stride=1, pad=False):
        m.update(word.encode())
    return tuple(m.digest())


def minhash_exists(minhash, minhashes):
    return minhash in minhashes


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


def pick_fragment(taxids, profiles, genome_db, profile_fun, similarity_fun,
                  max_its=100, nonalph_cutoff=None, **get_kwargs):
    """returns one randomly chosen fragment from the SeqRecords of the given taxids

    :param taxids: ids from which to choose, must have entries in genome_db
    :param profiles: set of objects to which the result of profile_fun is
                     compared
    :param genome_db: database allowing access to the SeqRecord given a taxid
    :param profile_fun: function which creates a profile of a given sequence
    :param similarity_fun: function which compares the profile of the fragment
                           candidate to all other profiles. if True, fragment
                           will *not* be picked
    :param max_its: maximum number of attempts until giving up, can be None
    :param get_kwargs: kwargs to pass to get_fragment
    :returns: fragment and its kmer-profile; or None if none could be picked
    """
    its = 0
    fragment = None
    profile = None
    while fragment is None:
        its += 1
        if (max_its is not None and its > max_its):
            return None
        taxid = choice(taxids)
        record = genome_db[taxid]
        fragment = get_fragment(record, **get_kwargs)
        if fragment is None:
            continue
        if (nonalph_cutoff is not None):
            nonalph_perc = (
                len([letter for letter in fragment
                     if letter.upper() not in ALPHABET])
                / len(fragment))
            if (nonalph_perc > nonalph_cutoff):
                fragment = None
                profile = None
                continue
        profile = profile_fun(str(fragment.seq))
        if (similarity_fun(profile, profiles)):
            fragment = None
            profile = None
            continue
    return fragment, profile, taxid

def pick_fragment_nocomp(taxids, genome_db, max_its=100, nonalph_cutoff=None, **get_kwargs):
    its = 0
    fragment = None
    while fragment is None:
        its += 1
        if (max_its is not None and its > max_its):
            return None
        taxid = choice(taxids)
        record = genome_db[taxid]
        fragment = get_fragment(record, **get_kwargs)
        if fragment is None:
            continue
        if (nonalph_cutoff is not None):
            nonalph = 0
            for letter in fragment:
                if (letter not in ALPHABET_all):
                    nonalph += 1
            nonalph_perc = nonalph / len(fragment)
            if (nonalph_perc > nonalph_cutoff):
                fragment = None
                continue
    return fragment,  taxid



def get_sk_fragments(nr_fragments, orders_dict, genome_db,
                     profile_fun, similarity_fun,
                     max_its=None, order_max_its=None,
                     nonalph_cutoff=None, thread_nr=None,
                     **get_fragment_kwargs):
    """obtains specified number of fragments based on the given order taxids

    :param nr_fragments: number of fragments to obtain
    :param orders_dict: dictionary, mapping order taxids to species taxids
    :param genome_db: genomeDB containing genomes of the species taxids
    :param profile_fun: function which creates a profile of a given sequence
    :param similarity_fun: function which compares the profile of the fragment
                           candidate to all other profiles. if True, fragment
                           will *not* be picked
    :param max_its: maximum number of iterations until giving up
    :param order_max_its: maximum number of iterations per order
    :returns: fragments and kmer-profiles
    """
    def finished():
        if (max_its is not None and iterations >= max_its):
            logging.warning("maximum iterations reached!")
            return True
        return (len(fragments) >= nr_fragments)
    profiles = []
    fragments = []
    speciess = []
    iterations = 0
    pbar = tqdm(total=nr_fragments, position=thread_nr,
                desc=genome_db.name)
    while not finished():
        for order in sample(list(orders_dict), len(orders_dict)):
            if (finished()):
                break
            iterations += 1
            avail_taxids = [taxid for taxid in orders_dict[order]
                            if taxid in genome_db]
            if (len(avail_taxids)) == 0:
                continue
            picked = pick_fragment(avail_taxids, profiles, genome_db,
                                   profile_fun, similarity_fun,
                                   order_max_its, nonalph_cutoff,
                                   **get_fragment_kwargs)
            if (picked is None):
                continue
            fragment, profile, species = picked
            fragments.append(fragment)
            profiles.append(profile)
            speciess.append(species)
            pbar.update()
    pbar.close()
    print(f'{iterations} iterations on superkingdom level')
    return fragments, profiles, speciess


def get_sk_fragments_nocomp(nr_fragments, orders_dict, genome_db, order_max_its=None,
                            nonalph_cutoff=None, **get_fragment_kwargs):
    def finished():
        return (len(fragments) >= nr_fragments)
    fragments = []
    speciess = []
    iterations = 0
    logging.info('precaching taxids with available genomes for each order... ')
    avail_taxids = {}
    for order in tqdm(orders_dict):
        taxids = [taxid for taxid in orders_dict[order]
                  if taxid in genome_db]
        if len(taxids) > 0:
            avail_taxids[order] = taxids
    orders_keys = list(avail_taxids)
    logging.info('done.')
    pbar = tqdm(total=nr_fragments,
                desc=genome_db.name)
    while not finished():
        for order in sample(orders_keys, len(orders_keys)):
            if (finished()):
                break
            iterations += 1
            taxids = avail_taxids[order]
            picked = pick_fragment_nocomp(taxids, genome_db,
                                   order_max_its, nonalph_cutoff,
                                   **get_fragment_kwargs)
            if (picked is None):
                continue
            fragment,  species = picked
            fragments.append(fragment)
            speciess.append(species)
            pbar.update()
    pbar.close()
    print(f'{iterations} iterations on superkingdom level')
    return fragments, speciess


def load_genomes(genome_dir, sk, thr=16e9):
    def from_fastadir(sk_dir):
        fastas = [os.path.join(sk_dir, _.strip())
                  for _ in open(os.path.join(sk_dir, 'files.txt')).readlines()]
        return GenomeDB(fastas, os.path.join(sk_dir, 'mapping.tsv'), name=sk, size_thr=thr)

    def from_fasta(fasta_file, mapping):
        return GenomeDB(fasta_file, mapping, name=sk, size_thr=thr)

    logging.info(f'loading {sk} genomes')
    if (sk == 'Archaea'):
        return from_fastadir(
            os.path.join(genome_dir, 'Archaea'))
    elif (sk == 'Bacteria'):
        return from_fasta(
            os.path.join(genome_dir, 'Bacteria/full_genome_bacteria.fna'),
            os.path.join(genome_dir, 'mapping_Bacteria.tsv'))
    elif (sk == 'Eukaryota'):
        return from_fastadir(
            os.path.join(genome_dir, 'Eukaryota'))
    elif (sk == 'Viruses'):
        return from_fasta(
            os.path.join(genome_dir, 'Viruses/all_viruses_db.fa'),
            os.path.join(genome_dir, 'mapping_Viruses.tsv'))
    else:
        raise Exception(f'genomes of {sk} not available')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sk')
    parser.add_argument('nr_fragments', type=int)
    parser.add_argument('--outdir', '-o', default='.')
    parser.add_argument('--thr', '-t', type=float, default=16e9)
    parser.add_argument('--nonalph_cutoff', type=float, default=None)
    parser.add_argument('--no_comp', action='store_true')
    parser.add_argument('--genome_dir', default='genomes/')
    parser.add_argument('--sk_order_dict', default='sk_order_dict.json')
    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    nr_seqs = args.nr_fragments
    sk = args.sk
    sk_order_dict = json.load(
        open(args.sk_order_dict))

    def minhash_defined(seq):
        return minhash(seq, 6, 6)

    def fake_profile_fun(seq):
        global counter
        counter += 1
        return counter

    def sk_fragments_plus_stats(sk, outdir='.', thread_nr=None):
        genome_db = load_genomes(
            args.genome_dir, sk, thr=args.thr)
        if (args.no_comp):
            global counter
            counter = 0
            fragments, profiles, speciess = get_sk_fragments_nocomp(
                nr_fragments=nr_seqs, orders_dict=sk_order_dict[sk], genome_db=genome_db,
                nonalph_cutoff=args.nonalph_cutoff)
        else:
            fragments, profiles, speciess = get_sk_fragments(
                nr_seqs, sk_order_dict[sk], genome_db, minhash_defined,
                minhash_exists, 10_000_000, None, args.nonalph_cutoff, thread_nr)
        print(
            f'{len(fragments)} fragments generated, alongside {len(profiles)} '
            'unique profiles')
        json.dump([str(seq.seq) for seq in fragments],
                  open(os.path.join(outdir, f'{sk}_fragments.json'), 'w'))
        with open(os.path.join(outdir, f'{sk}_species_picked.txt'), 'w') as f:
            f.writelines(str(sp) + '\n' for sp in speciess)
    logging.info(f'generating fragments for {sk}')
    sk_fragments_plus_stats(sk, args.outdir)
    logging.info(f'[{sk}] done')
