import numpy as np
import json
import os
import argparse
from analysis_atom import *
from analysis_profile import locate_interface, find_condensed_phase_edges


def wrap_z(pos, L):
    return pos - np.floor(pos / L) * L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input .lammpstrj file")
    parser.add_argument("phase", type=str, help="input .json file")
    parser.add_argument("output", type=str, help="output .npy file")
    parser.add_argument(
        "--label", type=str, default="", help=" .json file containing phase labels"
    )
    clargs = parser.parse_args()

    with open(clargs.phase, "r") as p:
        phases = json.load(p)
    phis = []
    for i in phases["phases"].keys():
        if int(i) > 0:
            phis.append(ref_compositions(phases["phases"][str(i)], phases))
    print(np.array(phis))

    phase_label_map = {}
    if len(clargs.label) > 0:
        with open(clargs.label, "r") as pp:
            phase_label_map = json.load(pp)

    Rg = 1.75

    mol_sequence_map = None
    all_data = []
    for frame in read_traj(clargs.input):
        try:
            atom_data = frame["atom"]
            if not mol_sequence_map:
                mol_sequence_map = map_Mol_Sequence(atom_data, phases["components"])
            Ls = [side[1] - side[0] for side in frame["box"]]
            bins = np.arange(0, Ls[-1], 2 * Rg)
            mid_bins = (bins[:-1] + bins[1:]) / 2.0
            for i in range(len(atom_data)):
                atom_data[i][-3:] = wrap(atom_data[i][-3:], Ls)

            com = get_CoM(atom_data[:, -3:])
            shift = Ls[-1] / 2.0 - com[-1]
            for i in range(len(atom_data)):
                atom_data[i][-1] += shift
                atom_data[i][-1] = wrap_z(atom_data[i][-1:], Ls[-1])

            profile = bin_data(
                atom_data,
                Ls,
                phis,
                mol_sequence_map,
                phases["N"],
                phase_label_map,
                bins,
            )
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
            mols_in_bulk = set([atom[1] for atom in atom_data[condensed_region]])
            num_gas = [0] * int(phases["N"])
            num_bulk = [0] * int(phases["N"])
            for mol in mols_in_gas:
                num_gas[mol_sequence_map[mol] - 1] += 1
            for mol in mols_in_bulk:
                num_bulk[mol_sequence_map[mol] - 1] += 1

            to_write = [left, right, 2 * Rg] + num_gas + num_bulk
            all_data.append(to_write)
        except:
            break

    np.save(clargs.output, np.array(all_data))
