import math, argparse, random
import numpy as np
import os
import gzip, pickle


def read_atom_file(f, sorted_mol=False):
    data = {}
    flag_time, flag_n, flag_box, pos = None, None, None, None
    atom_data = []
    box = []
    natoms = None
    for line in f:
        contents = line.split()
        if "TIMESTEP" in contents:
            flag_time = True
            continue
        if "NUMBER" in contents:
            flag_n = True
            continue
        if "BOX" in contents:
            flag_box = True
            continue
        if "mol" in contents and "type" in contents:
            pos = True
            continue
        if flag_time:
            data["timestep"] = int(contents[0])
            flag_time = False
        if flag_n:
            natoms = int(contents[0])
            flag_n = False
        if flag_box and len(box) < 3:
            box.append([float(s) for s in contents])
        if pos:
            atom_data.append([float(s) for s in contents[:]])
        if natoms and len(atom_data) == natoms:
            break
    data["atom"] = np.array(atom_data)
    data["box"] = box
    data["mol"] = None
    if sorted_mol:
        mol_labels = np.sort(np.unique((atom_data[:, 1])))
        nmols = len(mol_lables)
        mol_data = {}
        for n in range(nmols):
            mol_pos = atom_data[atom_data[:, 1] == mol_labels[n]]
            mol_data[mol_labels[n]] = mol_pos[
                mol_pos[:, 2].argsort()
            ]  # sort based on atom types
        data["mol"] = mol_data

    return data


def read_traj(traj_path):
    with open(traj_path, "r") as traj_file:
        while True:
            try:
                data = read_atom_file(traj_file)
                yield data
            except EOFError:
                break


def read_traj_zipped(traj_path):
    with gzip.open(traj_path, "rt") as traj_file:
        while True:
            try:
                data = read_atom_file(traj_file)
                yield data
            except EOFError:
                break


def groupby_molID(atom_data):
    mol = {}
    for atom in atom_data:
        if atom[1] not in mol:
            mol[atom[1]] = [list(atom)]
        else:
            mol[atom[1]].append(list(atom))
    return mol


def get_SeqType(atom_data, atom_ids, sequence_map):
    """ """
    sorted_atoms = sorted(np.array(atom_ids), key=lambda x: x[0])
    chain = [atom[2] for atom in sorted_atoms]
    pattern = "".join([str(int(t)) for t in chain])
    return sequence_map[pattern]


def map_Mol_Sequence(atom_data, sequence_map):
    res = {}
    mol = groupby_molID(atom_data)
    for mol_id in mol.keys():
        chain = mol[mol_id]
        res[mol_id] = get_SeqType(atom_data, chain, sequence_map)
    return res


def wrap(pos, Ls):
    #    ans = []
    #    for i in range(len(pos)):
    #        ans.append(pos[i] - np.floor(pos[i] / Ls[i]) * Ls[i])
    #    return ans
    return np.array(pos) - np.floor(np.array(pos) / np.array(Ls)) * np.array(Ls)


def unwrap(pos, Lz):
    return pos - np.floor(pos / Lz) * Lz


def unwrap_indx(index, mid):
    return list(index[mid:]) + list(index[:mid])


def get_CoM(mol_data):
    assert np.array(mol_data).shape[1] == 3
    return np.average(np.array(mol_data), axis=0)


def get_shiftAtoms(data, dshift, Ls):
    shifted_atoms = np.array(data)
    for i in range(data.shape[0]):
        shifted_atoms[i, -3:] = wrap(data[i, -3:] + dshift, Ls)
    return shifted_atoms


def ref_compositions(seqs, phase_description):
    sequences = phase_description["components"]
    tot_compo = {x + 1: 0 for x in range(phase_description["N"])}
    for seq in seqs.keys():
        nchain = seqs[seq]
        type_id = sequences[seq]
        tot_compo[int(type_id)] += nchain
    compo_vector = []
    for i in sorted(tot_compo.keys()):
        compo_vector.append(tot_compo[i])
    return np.array(compo_vector) / sum(compo_vector)


def density_filter(x, threshold=0.1):
    return np.heaviside(np.array(x) - threshold, 1)


def histogram_intersection(h1, h2):
    assert len(h1) == len(h2)
    return sum([min(h1[i], h2[i]) for i in range(len(h1))])


def histogram_divergence(h1, h2):
    assert len(h1) == len(h2)
    return (
        sum(
            [
                (h1[i] - h2[i]) ** 2 / (h1[i] + h2[i]) if h1[i] + h2[i] > 0 else 0
                for i in range(len(h1))
            ]
        )
        / 2
    )


def compute_degree_of_mixing(op1, op2, density, threshold):
    """based on chain id"""
    f, g = np.array(op1), np.array(op2)
    f = f / np.sum(f)
    g = g / np.sum(g)
    return histogram_intersection(f, g), histogram_divergence(f, g)


def bin_data(atom_data, Ls, phis, mol_sequence_map, N, phase_map, bins):
    """
    atom_data:  a dictionary with the format as if a .atom file has been read in through the read_atom_file
    routine
    phis: a list contains K reference vectors for ideal phase compositions
    """
    rhos, ops, compos, labels, nchains = [], [], [], [], []
    res = {}
    for i in range(len(bins) - 1):
        zmin, zmax = bins[i], bins[i + 1]
        chunks = np.array(atom_data)
        chunks = chunks[chunks[:, -1] < zmax]
        chunks = chunks[chunks[:, -1] >= zmin]

        compositions = np.zeros(N)  # chain compositions
        chain_ids = set()
        v0 = np.pi / 6
        rho = v0 * len(chunks) / (Ls[0] * Ls[1] * (zmax - zmin))

        for atoms in chunks:
            chain_ids.add(atoms[1])
        for chain in chain_ids:  # FIXME standardize the chain type label index
            chain_type = mol_sequence_map[chain]  # component type: 1, ..., N
            compositions[chain_type - 1] += 1

        nchains.append(compositions)
        if np.sum(compositions) > 0.0:
            compositions = compositions / np.sum(compositions)
        # order parameter for K phases
        op_local = np.zeros(len(phis))
        for k in range(len(phis)):
            if np.linalg.norm(compositions, ord=2) > 0:
                op_local[k] = (
                    np.dot(compositions, phis[k])
                    / np.linalg.norm(compositions, ord=2)
                    / np.linalg.norm(phis[k], ord=2)
                )
        if len(phase_map) > 0:
            ps = np.zeros(len(set(val for val in phase_map.values())))
            # alpha-phase beta-phase and dilute phase at coexistence
            # only interested in the degree of mixing for the alpha and beta phases
            for chain_id in list(chain_ids):
                label = phase_map[str(int(chain_id))]
                ps[label] += 1
            labels.append(ps)

        ops.append(op_local)
        rhos.append(rho)
        compos.append(compositions)

    res["op"] = np.array(ops)
    res["chain_composition"] = np.array(compos)
    if len(labels) > 0:
        res["label"] = np.array(labels)
    else:
        res["label"] = None
    res["density"] = np.array(rhos)
    res["bins"] = bins
    res["nchains"] = np.array(nchains)
    # res['dom'] = compute_degree_of_mixing(ops[:,2], ops[:,3], rhos, 0.1)

    return res
