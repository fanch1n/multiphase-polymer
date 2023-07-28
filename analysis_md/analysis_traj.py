import math, argparse, random
import numpy as np
from itertools import product
import json
import os
from collections import Counter

# from analysis_lammps import groupby_molID, map_Mol_Sequence, get_SeqType
from analysis_atom import *  # groupby_molID, map_Mol_Sequence, get_SeqType


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


def compute_degree_of_mixing(op1, op2):
    f, g = np.array(op1), np.array(op2)
    f = f / np.sum(f)
    g = g / np.sum(g)
    return 1.0 - histogram_divergence(f, g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=str, help="input .json file containing phase descriptions"
    )
    parser.add_argument("labels", type=str, help=".json file containing phase labels")
    parser.add_argument(
        "trajlist", nargs="+", help="dump.lammpstrj file for processing"
    )
    parser.add_argument("alpha", type=int, help="label index for alpha phase")
    parser.add_argument("beta", type=int, help="label index for beta phase")
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="frequency for processing the trajectory [10]",
    )
    parser.add_argument("--output", type=str, help="prefix for output files")
    parser.add_argument(
        "--makeplots",
        action="store_true",
        help="plot the time series to output.pdf [False]",
    )
    clargs = parser.parse_args()

    with open(clargs.input, "r") as p:
        phases = json.load(p)
    with open(clargs.labels, "r") as g:
        # phase_map currently only contains two labels
        phase_map = json.load(g)
    phis = []
    for i in phases["phases"].keys():
        if i != "0":
            phis.append(ref_compositions(phases["phases"][str(i)], phases))
    print("target condensed phases:\n", np.array(phis))

    mol_sequence_map = None
    trajfiles = clargs.trajlist
    list_dom = []
    list_timestep = []
    counter = 0
    for traj in trajfiles:
        for frame in read_traj(traj):
            try:
                if counter % clargs.freq == 0:
                    print(counter, flush=True)
                    if not mol_sequence_map:
                        mol_sequence_map = map_Mol_Sequence(
                            frame["atom"], phases["components"]
                        )
                        Ls = [side[1] - side[0] for side in frame["box"]]
                        bins = np.arange(0.0, Ls[-1], 3.0)
                        mid_bins = (bins[:-1] + bins[1:]) / 2.0
                    profile = bin_data(
                        frame["atom"],
                        Ls,
                        phis,
                        mol_sequence_map,
                        phases["N"],
                        phase_map,
                        bins,
                    )
                    dom = compute_degree_of_mixing(
                        profile["label"][:, 0], profile["label"][:, 1]
                    )
                    # print(dom)
                    list_dom.append(dom)
                    list_timestep.append(frame["timestep"])
                counter += 1

            except:
                print("break at ", counter)
                break

    #     # time series test for stationarity and estimation of correlation time
    if clargs.output is not None:
        output_name = "%s-%d%d.npy" % (clargs.output, clargs.alpha, clargs.beta)
        np.save(output_name, np.array([list_timestep, list_dom]))
