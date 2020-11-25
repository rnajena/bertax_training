import os
from collections import defaultdict


def get_dicts(data_dir="/mnt/fass2/projects/fm_read_classification_comparison/taxonomy"):
    """get ncbi data and parse to dicts
    parent_dict: dict returning parent of input id
    scientific_names: returning clade name of input id
    phylo_names: returning taxonomical rank of id
    """
    col_delimiter = '\t|\t'
    row_delimiter = '\t|\n'

    parent_dict = {}
    phylo_names = {}

    scientific_names = {}
    common_names = defaultdict(set)
    genbank_common_name = defaultdict(set)
    with open(os.path.join(data_dir, 'names.dmp')) as names_file:
        for line in names_file:
            line = line.rstrip(row_delimiter)
            values = line.split(col_delimiter)
            tax_id, name_txt, _, name_type = values[:4]
            if name_type == 'scientific name':
                scientific_names[int(tax_id)] = name_txt
            elif name_type == 'common name' or name_type == 'equivalent name' or name_type == 'synonym':
                common_names[int(tax_id)].add(name_txt)
            elif name_type == 'genbank common name':
                genbank_common_name[int(tax_id)].add(name_txt)

    with open(os.path.join(data_dir, 'nodes.dmp')) as nodes_file:
        for line in nodes_file:
            line = line.rstrip(row_delimiter)
            values = line.split(col_delimiter)
            tax_id, parent_id, phylo_rank = values[:3]
            tax_id = int(tax_id)
            parent_id = int(parent_id)
            parent_dict.update({tax_id: parent_id})
            phylo_names.update({tax_id: str(phylo_rank)})

    scientific_names_inv = {str(v): k for k, v in scientific_names.items()}
    common_names_inv = {i: k for k, v in common_names.items() for i in v}
    return parent_dict, scientific_names, common_names, phylo_names, genbank_common_name, scientific_names_inv, common_names_inv


def get_tax_path(id, parent_dict, phylo_names):
    current_id = int(id)
    taxonomy = [id]
    while current_id != 1 and phylo_names[current_id] != "superkingdom":
        current_id = parent_dict[current_id]
        taxonomy.append(current_id)
    return taxonomy


class TaxDB():

    def __init__(self, data_dir="/home/go96bix/projects/deep_eve/taxonomy"):
        self.data_dir = data_dir
        parent_dict, scientific_names, common_names, phylo_names, genbank_common_name, scientific_names_inv, common_names_inv = get_dicts(data_dir=data_dir)
        self.parent_dict = parent_dict
        self.scientific_names = scientific_names
        self.common_names = common_names
        self.phylo_names = phylo_names
        self.genbank_common_name =genbank_common_name
        self.scientific_names_inv = scientific_names_inv
        self.common_names_inv = common_names_inv

    def search_from_id(self, taxID):
        tax_path = get_tax_path(taxID, self.parent_dict, self.phylo_names)
        scientific_names_path = [self.scientific_names[i] for i in tax_path]
        common_names_path = [self.common_names[i] for i in tax_path]
        phylo_names_path = [self.phylo_names[i] for i in tax_path]
        genbank_common_names_path = [self.genbank_common_name[i] for i in tax_path]
        taxID_entry = TaxID_entry(tax_path,scientific_names_path,common_names_path,phylo_names_path,genbank_common_names_path)
        return taxID_entry

    def search_from_name(self, name):
        try:
            taxID = self.scientific_names_inv[name]
        except:
            try:
                taxID = self.common_names_inv[name]
            except:
                print(f"{name} not found")
                return None
        return TaxDB.search_from_id(self,taxID)

class TaxID_entry():
    def __init__(self, tax_path, scientific_names_path, common_names_path, phylo_names_path, genbank_common_names_path):
        self.tax_path = tax_path
        self.scientific_names_path = scientific_names_path
        self.common_names_path = common_names_path
        self.phylo_names_path = phylo_names_path
        self.genbank_common_names_path = genbank_common_names_path


class TaxidLineage:
    def __init__(self):
        from ete3 import NCBITaxa
        self.ncbi = NCBITaxa()
        self.cache = {}

    def populate(self, taxids, ranks=['superkingdom', 'kingdom', 'phylum', 'family']):
        for taxid in taxids:
            d = self.ncbi.get_rank(self.ncbi.get_lineage(taxid))
            self.cache[taxid] = {r: self._get_d_rank(d, r) for r in ranks}

    def _get_d_rank(self, d, rank):
        if (rank not in d.values()):
            return (None, 'unknown')
        taxid = [k for k, v in d.items() if v == rank]
        name = self.ncbi.translate_to_names(taxid)[0]
        return (taxid[0], name if isinstance(name, str) else 'unknown')

    def get_ranks(self, taxid, ranks=['superkingdom', 'kingdom', 'phylum', 'family']):
        if taxid in self.cache:
            return self.cache[taxid]
        d = self.ncbi.get_rank(self.ncbi.get_lineage(taxid))
        return {r: self._get_d_rank(d, r) for r in ranks}





if __name__ == "__main__":
    parent_dict, scientific_names, common_names, phylo_names, genbank_common_name = get_dicts()
