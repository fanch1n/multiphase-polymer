import numpy as np

def read_atom_file(filename):
    '''
    read in .atom file
    assume the data had the format [atom_id mol_id type xu yu zu]
    '''
    data = {}
    with open(filename, 'r') as f:
        print("reading_traj_file input:", filename)
        cnt = 0
        atom_data = []
        for line in f:
            contents = line.split()
            if 'TIMESTEP' in contents:
                #if cnt > 0 and cnt % freq == 0:
                #    a = np.array(atom_data)
                #    Ls = np.array(box)[:, 1] - np.array(box)[:, 0]
                #    bins = np.arange(0, Ls[-1], 3)
                cnt += 1
                nchains = 0
                atom_data = []
                box = []
                pos = False
                bx = False
                newstep = True
                natoms = False
            if newstep and not bx and not natoms:
                if len(contents) == 1:
                    timestep = int(contents[0])
            if 'BOX' in contents:
                bx = True
            elif 'type' in contents:
                pos = True
                bx = False
            else:
                if len(contents) > 0 and bx:
                    box.append([float(s) for s in contents])
                if len(contents) > 0 and pos:
                    atom_id, mol_id = int(contents[0]), int(contents[1])
                    info = [atom_id, mol_id] + [float(s) for s in contents[2:]]
                    nchains = max(nchains, mol_id)
                    atom_data.append(info)
    data['atom'] = np.array(atom_data)
    data['box'] = box
    data['n_mol'] = nchains

    return data


def groupby_molID(atom_data):
    mol = {}
    for atom in atom_data:
        if atom[1] not in mol:
            mol[atom[1]] = [list(atom)]
        else:
            mol[atom[1]].append(list(atom))
    return mol


def get_SeqType(atom_data, atom_ids, sequence_map):
    '''
    '''
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

def unwrap(pos, Lz):
    return pos - np.floor(pos / Lz) * Lz

def unwrap_indx(index, mid):
    return list(index[mid:]) + list(index[:mid])

def get_CoM(mol_data):
    assert np.array(mol_data).shape[1] == 3
    return np.average(np.array(mol_data), axis=0)


