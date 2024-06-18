import argparse
import numpy as np
import json
from itertools import product, combinations

from analysis_traj import ref_compositions
from analysis_atom import *
from analysis_profile import find_condensed_phase_edges, locate_condense_phase


def shift_center(atom_data, Ls, phis, mol_sequence_map, ntypes, phase_map):
    # an additional step to center the whole box
    c_atom_data = np.array(atom_data)
    c_atom_data[:, -3:] = wrap(c_atom_data[:, -3:], Ls)
    net_CoM = get_CoM(c_atom_data[:, -3:])
    c_atom_data = get_shiftAtoms(
        c_atom_data, np.array([0, 0, Ls[-1] / 2.0]) - net_CoM, Ls
    )
    profile = bin_data(c_atom_data, Ls, phis, mol_sequence_map, ntypes, phase_map, bins)
    left_edge, right_edge = find_condensed_phase_edges(profile["density"], bins)
    dz = Ls[-1] / 2.0 - (right_edge + left_edge) / 2.0

    return get_shiftAtoms(c_atom_data, np.array([0.0, 0.0, dz]), Ls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input .atom file or .lammpstrj file")
    parser.add_argument(
        "phase", type=str, help="input .json file containing phase information"
    )
    parser.add_argument(
        "--label", type=str, default="", help=" .json file containing phase labels"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the processed profile [False]",
    )
    parser.add_argument("--output", type=str, help="output .svg file")
    clargs = parser.parse_args()

    with open(clargs.phase, "r") as p:
        phases = json.load(p)

    phase_label_map = {}
    if len(clargs.label) > 0:
        with open(clargs.label, "r") as pp:
            phase_label_map = json.load(pp)

    phis = []
    for i in phases["phases"].keys():
        if i != "0":
            phis.append(ref_compositions(phases["phases"][str(i)], phases))
    print(np.array(phis))

    string = clargs.input.split("/")[-1].split("-")[-1].split(".")[0]
    alpha, beta = int(string[0]), int(string[1])
    print("coexistence of %d-%d" % (alpha, beta))

    list_compos = []
    list_ops = []
    freq = 100
    counter = 0

    read_func = None
    if ".gz" in clargs.input:
        read_func = read_traj_zipped
    else:
        read_func = read_traj

    coex_labels = [alpha - 1, beta - 1]
    # track the counts of different chain types in phase-alpha and phase-beta
    chain_counts = [[] for _ in coex_labels]
    for frame in read_func(clargs.input):
        try:
            if counter == 0:
                atom_data = frame["atom"]
                Ls = [side[1] - side[0] for side in frame["box"]]
                bins = np.arange(0.0, Ls[-1], 3.0)
                mid_bins = (bins[:-1] + bins[1:]) / 2.0
                mol_sequence_map = map_Mol_Sequence(
                    np.array(atom_data), phases["components"]
                )

            if counter % freq == 0:
                shifted_atoms = shift_center(
                    frame["atom"],
                    Ls,
                    phis,
                    mol_sequence_map,
                    phases["N"],
                    phase_label_map,
                )
                profile = bin_data(
                    shifted_atoms,
                    Ls,
                    phis,
                    mol_sequence_map,
                    phases["N"],
                    phase_label_map,
                    bins,
                )
                list_compos.append(profile["chain_composition"])
                list_ops.append(profile["op"])

                # alpha-phase
                for i in range(len(coex_labels)):
                    bulk_loc = locate_condense_phase(profile["op"][:, coex_labels[i]])
                    print("phase-%d" % (coex_labels[i] + 1), bulk_loc)
                    collect = []
                    for region in bulk_loc:
                        # chain_counts[i].append(profile["nchains"][region[0]+1:region[1]-1])
                        collect.append(
                            profile["nchains"][region[0] + 1 : region[1] - 1]
                        )
                    chain_counts[i].append(collect)
                print(counter, flush=True)

            counter += 1
        except:
            print("break at ", counter)
            break

    if clargs.save:
        np.save("proc-%d%d-compo.npy" % (alpha, beta), np.array(list_compos))
        np.save("proc-%d%d-ops.npy" % (alpha, beta), np.array(list_ops))
        np.save(
            "proc-%d%d-chains.npy" % (alpha, beta), np.array(chain_counts, dtype=object)
        )
