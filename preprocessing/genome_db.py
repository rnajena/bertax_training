from os.path import getsize, basename
from Bio import SeqIO
from random import choice
from logging import info
from time import time
import threading


class GenomeDB:
    """has members:
    seq_dict: SeqIO dictionary accessible by accession
    spec_dict: dictionary mapping species taxids to list of accessions
    """
    def __init__(self, fasta_or_fastas, mapping=None, size_thr=16e9,
                 name='untitled'):
        """creates database of sequences accessible by taxid

        :param fasta_or_fastas: either path to a multifasta or paths
                                to multiple fastas
        :param str mapping: file containing mapping of IDs to taxids
                            if None, assumes that IDs are already taxids
        :param size_thr: cumulative file size below which sequences are
                         stored in memory
        """
        def key_function(id_):
            # use accession(ref) or whole id
            if (isinstance(id_, SeqIO.SeqRecord)):
                id_ = id_.id
            ids = id_.split('|')
            if len(ids) < 4 or ids[2] != 'ref':
                return id_
            else:
                return ids[3]
        self.name = name
        self.spec_dict = (self.read_mapping(mapping)
                          if mapping is not None
                          else None)
        fastas = ([fasta_or_fastas] if isinstance(fasta_or_fastas, str)
                  else fasta_or_fastas)
        info('determining total size of fasta(s)...')
        total_size = sum([getsize(fasta) for fasta in fastas])
        if (total_size > size_thr):
            info(f'size {total_size/1024**3:.2f}GB above threshold '
                 f'({size_thr/1024**3:.2f}GB), creating index')
            start = time()
            self.seq_dict = SeqIO.index_db(basename(fastas[0]) + f'_{threading.get_ident()}.idx',
                                           fastas,
                                           format='fasta',
                                           key_function=key_function)
            info(f'index created in {time() - start:.2f} seconds')
        else:
            info(f'size {total_size/1024**3:.2f}GB below threshold '
                 f'({size_thr/1024**3:.2f}GB), storing all sequences in '
                 'memory')
            start = time()
            self.seq_dict = dict(item for fasta in fastas
                                 for item in
                                 SeqIO.to_dict(
                                     SeqIO.parse(open(fasta), 'fasta'),
                                     key_function=key_function).items())
            info(f'sequence dictionary created in {time() - start:.2f} '
                 'seconds')

    def read_mapping(self, mapping_file):
        """reads a mapping from a file where each line has the form:
        <id> <taxid>

        :param str mapping_file: filepath to read from
        :return" dictionary mapping IDs to taxids
        """
        mapping = {}
        with open(mapping_file) as f:
            for line in f:
                line = line.strip().split()
                if (len(line) < 2):
                    continue
                mapping.setdefault(int(line[1]), []).append(line[0])
        return mapping

    def all_seqs(self, taxid):
        """returns all SeqRecords stored for the given species taxid

        :param taxid: species taxid
        :returns: associated SecRecords
        """
        if (self.spec_dict is None):
            return [self.seq_dict[str(taxid)]]
        else:
            return [self.seq_dict[id_] for id_ in self.spec_dict[taxid]]

    def __getitem__(self, idx):
        """returns a stored sequence for the given taxid

        NOTE: when more than one sequence is stored for a given taxid,
        one of these is chosen and returned randomly

        :param idx: species taxid
        :return: SeqRecord associated with taxid
        """
        seqids = (self.spec_dict[idx] if self.spec_dict is not None
                  else [str(idx)])
        return self.seq_dict[choice(seqids)]

    def __len__(self):
        """number of species with sequences stored
        NOTE: there might be more sequences than species!

        :returns: number of species
        """
        return len(self.spec_dict if self.spec_dict is not None
                   else self.seq_dict)

    def __iter__(self):
        """iterate over species taxids"""
        return iter(self.spec_dict)
