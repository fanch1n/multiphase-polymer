import math, argparse, random
import numpy as np
from itertools import product
import json
import os
from collections import Counter

# def read_atom_file(filename):
#    '''
#    read in .atom file
#    assume the data had the format [atom_id mol_id type xu yu zu]
#    '''
#    data = {}
#    with open(filename, 'r') as f:
#        print("reading_traj_file input:", filename)
#        cnt = 0
#        atom_data = []
#        for line in f:
#            contents = line.split()
#            if 'TIMESTEP' in contents:
#                cnt += 1
#                nchains = 0
#                atom_data = []
#                box = []
#                pos = False
#                bx = False
#                newstep = True
#                natoms = False
#            if newstep and not bx and not natoms:
#                if len(contents) == 1:
#                    timestep = int(contents[0])
#            if 'BOX' in contents:
#                bx = True
#            elif 'type' in contents:
#                pos = True
#                bx = False
#            else:
#                if len(contents) > 0 and bx:
#                    box.append([float(s) for s in contents])
#                if len(contents) > 0 and pos:
#                    atom_id, mol_id = int(contents[0]), int(contents[1])
#                    info = [atom_id, mol_id] + [float(s) for s in contents[2:]]
#                    nchains = max(nchains, mol_id)
#                    atom_data.append(info)
#    data['atom'] = np.array(atom_data)
#    data['box'] = box
#    data['n_mol'] = nchains
#
#    return data


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
    ans = []
    for i in range(len(pos)):
        ans.append(pos[i] - np.floor(pos[i] / Ls[i]) * Ls[i])
    return ans


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


def bin_data(atom_data, Ls, phis, mol_sequence_map, N, chain_id_divider, bins):
    """
    atom_data:  a dictionary with the format as if a .atom file has been read in through the read_atom_file
    routine
    phis: a list contains K reference vectors for ideal phase compositions
    """
    rhos, ops, compos, labels = [], [], [], []
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

        # initial phase label #FIXME implement this in a more general way?
        p1, p2 = 0, 0
        for chain_id in list(chain_ids):
            # if chain_id > 2 * chain_id_divider:
            # print("incorrect phase label flag")
            # exit(1)
            if chain_id <= chain_id_divider:
                p1 += 1
            elif chain_id <= 2 * chain_id_divider:
                p2 += 1

        ops.append(op_local)
        labels.append([p1, p2])
        rhos.append(rho)
        compos.append(compositions)

    res["op"] = np.array(ops)
    res["chain_composition"] = np.array(compos)
    res["label"] = np.array(labels)
    res["density"] = np.array(rhos)
    res["bins"] = bins
    # res['dom'] = compute_degree_of_mixing(ops[:,2], ops[:,3], rhos, 0.1)

    return res
