import math, argparse, random
import numpy as np
from itertools import product
import json
import os
from collections import Counter
from analysis_lammps import groupby_molID, map_Mol_Sequence, get_SeqType


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


def read_traj_file(
    filename, phase_description, alpha, beta, freq, chain_id_divider, output
):
    """read in .lammpstrj file line by line
    assume the data had the format [atom_id mol_id type xu yu zu]
    organize data from every step by np array sorted by atom_id"""
    pos = False
    bx = False
    natoms = False
    newstep = False
    timestep = numatoms = None
    frame = []
    rawcompo = []
    density = []
    mixing = []
    nchains = 0
    cnt = 0
    mol_seq_map = None

    print("reading_traj_file input:", filename)
    with open(filename, "r") as f:
        for line in f:
            contents = line.split()
            if "TIMESTEP" in contents:
                if cnt > 0 and cnt % freq == 0:
                    a = np.array(atom_data)
                    Ls = np.array(box)[:, 1] - np.array(box)[:, 0]
                    bins = np.arange(0, Ls[-1], 3)
                    if not mol_seq_map:
                        mol_seq_map = map_Mol_Sequence(a, phase_description)
                    # compute profile per frame
                    ops, rhos, compos, dom = bin_data(
                        a,
                        phase_description,
                        alpha,
                        beta,
                        mol_seq_map,
                        chain_id_divider,
                        bins,
                    )

                    frame.append(ops)
                    density.append(rhos)
                    mixing.append(dom)
                    rawcompo.append(compos)

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

    if output:
        np.save(output, np.array(frame))
        filepath, filename = "/".join(output.split("/")[:-1]), output.split("/")[-1]
        np.save(os.path.join(filepath, "compo-" + filename), np.array(rawcompo))
        np.save(os.path.join(filepath, "mixing-" + filename), np.array(mixing))
        np.save(os.path.join(filepath, "density-" + filename), np.array(density))

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=str, help="input .json file containing phase descriptions"
    )
    parser.add_argument("alpha", type=int, help="label index for alpha phase")
    parser.add_argument("beta", type=int, help="label index for beta phase")
    parser.add_argument("trajfile", type=str, help="dump.lammpstrj file for processing")
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="frequency for processing the trajectory [10]",
    )
    parser.add_argument(
        "--id-divider",
        type=int,
        default=0,
        help="mol_id < id-divider phase alpha else phase beta [0]",
    )
    parser.add_argument("--output", type=str, help="prefix for output files")
    parser.add_argument(
        "--makeplots",
        action="store_true",
        help="plot the time series to output.pdf [False]",
    )
    clargs = parser.parse_args()

    if clargs.output is not None:
        output_name = "%s-%d%d-proc.npy" % (clargs.output, clargs.alpha, clargs.beta)
    else:
        output_name = None

    with open(clargs.input, "r") as p:
        phases = json.load(p)

    descript = []
    for i in phases["phases"].keys():
        descript.append(ref_compositions(phases["phases"][str(i)], phases))
    print(np.array(descript))

    if clargs.id_divider == 0:
        chain_id_divider = sum(
            [
                phase_description["phases"][str(alpha)][key]
                for key in phase_description["phases"][str(alpha)]
            ]
        )
    else:
        chain_id_divider = clargs.id_divider

    frame = read_traj_file(
        clargs.trajfile,
        phases,
        clargs.alpha,
        clargs.beta,
        clargs.freq,
        chain_id_divider,
        output_name,
    )
