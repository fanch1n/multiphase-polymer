import numpy as np
import json
import os
import argparse
from analysis_atom import wrap, get_CoM
from analysis_lammps import read_atom_file, map_Mol_Sequence
from analysis_atom import bin_data, ref_compositions
from analysis_profile import locate_interface, find_condensed_phase_edges


def wrap_z(pos, L):
    return pos - np.floor(pos / L) * L


def process_traj(filename, phases, phis):
    """read in .lammpstrj file line by line
    assume the data had the format [atom_id mol_id type xu yu zu]
    organize data from every step by np array sorted by atom_id"""
    pos = False
    bx = False
    natoms = False
    newstep = False
    timestep = None
    nchains = 0
    cnt = 0
    print("reading_traj_file input:", filename)
    all_data = []

    mol_sequence_map = None
    with open(filename, "r") as f:
        for line in f:
            contents = line.split()
            if "TIMESTEP" in contents:
                if cnt > 0:
                    atom_data = np.array(atom_data)

                    if not mol_sequence_map:
                        mol_sequence_map = map_Mol_Sequence(
                            atom_data, phases["components"]
                        )
                    # process the current frame
                    # raw_data = data['atom']
                    Ls = [side[1] - side[0] for side in box]

                    for i in range(len(atom_data)):
                        atom_data[i][-3:] = wrap(atom_data[i][-3:], Ls)

                    com = get_CoM(atom_data[:, -3:])
                    shift = Ls[-1] / 2.0 - com[-1]
                    # print("CoM of all atoms: ", com) # print("shift z by: ", shift)
                    for i in range(len(atom_data)):
                        atom_data[i][-1] += shift
                        atom_data[i][-1] = wrap_z(atom_data[i][-1:], Ls[-1])

                    # mol_sequence_map = map_Mol_Sequence(np.array(atom_data), phases['components'])
                    bins = np.arange(0, Ls[-1], 3.0)
                    mid_bins = (bins[:-1] + bins[1:]) / 2.0

                    profile = bin_data(
                        atom_data, Ls, phis, mol_sequence_map, phases["N"], 432, bins
                    )

                    # profile = bin_data(atom_data, Ls, phis, mol_sequence_map, phases['N'], 432, bins)
                    Rg = 1.6

                    density_profile = np.array(profile["density"])
                    max_density = max(density_profile)
                    threshold = max_density / 5.0
                    left, right = find_condensed_phase_edges(density_profile, bins)

                    assert left < right, "left edge must be smaller than right edge"
                    assert (
                        np.mean(
                            density_profile[
                                np.logical_and((left < mid_bins), (mid_bins < right))
                            ]
                        )
                        > threshold
                    ), "condensed phase is in the middle"
                    condensed_region = np.logical_and(
                        (atom_data[:, -1] > left), (atom_data[:, -1] < right)
                    )
                    gas_region = np.logical_or(
                        (atom_data[:, -1] < left - 2 * Rg),
                        (atom_data[:, -1] > right + 2 * Rg),
                    )
                    mols_in_gas = set([atom[1] for atom in atom_data[gas_region]])
                    mols_in_bulk = set(
                        [atom[1] for atom in atom_data[condensed_region]]
                    )
                    num_gas = [0] * int(phases["N"])
                    num_bulk = [0] * int(phases["N"])

                    for mol in mols_in_gas:
                        num_gas[mol_sequence_map[mol] - 1] += 1

                    for mol in mols_in_bulk:
                        num_bulk[mol_sequence_map[mol] - 1] += 1

                    to_write = [left, right, 2 * Rg] + num_gas + num_bulk
                    print(to_write)
                    all_data.append(to_write)

                cnt += 1
                nchains = 0
                atom_data = []
                box = []
                res = {}
                pos = False
                bx = False
                newstep = True
                natoms = False

            if newstep and not bx and not natoms:
                if len(contents) == 1:
                    timestep = int(contents[0])

            if "BOX" in contents:
                bx = True
            elif "type" in contents:
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

    return np.array(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input .lammpstrj file")
    parser.add_argument("phase", type=str, help="input .json file")
    parser.add_argument("--output", type=str, help="output .npy file")
    clargs = parser.parse_args()

    # dpath='/Users/fanc/della-home/GS/polymer-test/03_04_pipeline_N3/N4-index4/comparison/mineig0.5/correct-gas/diag-100.0-7/dilute-coex-0.30'
    # a, b = 0, 2
    # file='final-stitch-%d%d.lammpstrj' %(a, b)
    # phase_file = '/Users/fanc/della-home/GS/polymer-test/03_04_pipeline_N3/N4-index4/comparison/mineig0.5/correct-gas/diag-100.0-7/phase.json'

    # # data = read_atom_file(os.path.join(dpath, file)) #FIXME add an option for wrapping coordinates
    # datafile = os.path.join(dpath, file) #FIXME add an option for wrapping coordinates

    # Ls = [side[1]-side[0] for side in data['box']]

    with open(clargs.phase, "r") as p:
        phases = json.load(p)
    phis = []
    for i in phases["phases"].keys():
        phis.append(ref_compositions(phases["phases"][str(i)], phases))
    print(np.array(phis))

    collect = process_traj(clargs.input, phases, phis)
    np.save(clargs.output, collect)
