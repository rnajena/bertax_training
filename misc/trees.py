from ete3 import Tree, NCBITaxa, TreeStyle, NodeStyle, faces, AttrFace, CircleFace, COLOR_SCHEMES
from collections import defaultdict
from pprint import pprint
from os.path import basename, splitext


freq_style = NodeStyle({'vt_line_color': '#0000ff',
                        'hz_line_color': '#0000ff',
                        'vt_line_width': 3,
                        'hz_line_width': 3})
normal_style = NodeStyle({'vt_line_style': 1,
                          'hz_line_style': 1})

ncbi = NCBITaxa()


def get_sk_tree(taxids):
    tree = ncbi.get_topology(taxids, True, 'family', True)
    print(tree.get_ascii(attributes=['sci_name']))

    # get rid of unclassified virus /species/
    for c in tree.traverse():
        if c.rank == 'species':
            c.detach()
        elif c.rank == 'genus':
            # parent ≅ family
            c.detach()
    return tree


def layout(node):
    global highest_freq
    global freqs
    F = faces.TextFace(node.sci_name, tight_text=True,
                       fgcolor=('black' if not node.rank == 'order'
                                else 'green'))
    faces.add_face_to_node(F, node, column=0,
                           position="branch-right")
    if (not node.is_root()):
        freq = freqs[node.taxid]
        if (freq != 0):
            node.set_style(freq_style)
            C = CircleFace(radius=100 * (freq/highest_freq),
                           color='Blue', style='sphere')
            nr_parent = freqs[node.up.taxid]

            if (nr_parent != 0):
                perc = f'{freq / nr_parent * 100:.0f}% ({freq / freqs[1] * 100:.0f}%, {freq})'
            else:
                perc = 'na'
            F_perc = faces.TextFace(perc, tight_text=True)
            C.opacity = 0.3
            faces.add_face_to_node(F_perc, node, 0, position='branch-top')
            faces.add_face_to_node(C, node, 1, position='float')
            if freq >= 10_000 and freq < 12_000:
                print(f'{node.sci_name} ({node.taxid}): {freq}')
        else:
            node.set_style(normal_style)
    else:
        node.set_style(freq_style)


def show_tree(tree, layout, save_as='tree.svg'):
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.branch_vertical_margin = 10
    ts.layout_fn = layout
    tree.render(save_as)
    tree.show(tree_style=ts)


# store #occurrences for each taxid in the species' lineages
def taxid_freqs(taxids):
    freqs = defaultdict(int)
    for taxid in taxids:
        for ltaxid in ncbi.get_lineage(taxid):
            freqs[ltaxid] += 1
    return freqs


if __name__ == '__main__':
    import sys
    taxids_file, mapping_file = sys.argv[1:]
    with open(mapping_file) as f:
        taxids = {int(line.split()[1].strip()) for line in f.readlines()}
    tree = get_sk_tree(taxids)

    picked_taxids = [int(line.strip()) for line in
                     open(taxids_file).readlines()
                     if len(line.strip()) > 0]
    freqs = taxid_freqs(picked_taxids)
    highest_freq = list(sorted(freqs.values()))[-1]
    show_tree(tree, layout, save_as=splitext(basename(taxids_file))[0] + '.svg')
