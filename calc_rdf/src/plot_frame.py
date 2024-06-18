import math, argparse, random
import numpy as np
from itertools import product
import json
import os
from collections import Counter

from test import ref_compositions
from test import read_traj_file
from test import bin_data


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


def read_atom_file(filename):
    with open(filename, "r") as f:
        print("reading_traj_file input:", filename)
        cnt = 0
        atom_data = []
        for line in f:
            contents = line.split()
            if "TIMESTEP" in contents:
                if cnt > 0 and cnt % freq == 0:
                    a = np.array(atom_data)
                    Ls = np.array(box)[:, 1] - np.array(box)[:, 0]
                    bins = np.arange(0, Ls[-1], 3)
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
        return atom_data


def map_Mol_Sequence(atom_data, phase_description):
    res = {}
    mol = groupby_molID(atom_data)
    sequences = phase_description["components"]
    for mol_id in mol.keys():
        chain = mol[mol_id]
        res[mol_id] = getType(atom_data, chain, sequences)
    return res


def wrap(pos, Ls):
    ans = []
    for i in range(len(pos)):
        ans.append(pos[i] - np.floor(pos[i] / Ls[i]) * Ls[i])
    return ans


def plot_config(atom_data, fig, tot, i, style=None):
    ax = fig.add_subplot(1, tot, i, projection="3d")
    xr = [0, 20]
    yr = [0, 20]
    zr = [0, 120]
    diff = [xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]]
    for s, e in combinations(np.array(list(product(xr, yr, zr))), 2):
        if np.sum(np.abs(s - e)) in diff:
            ax.plot3D(*zip(s, e), color="black")
    Ls = [20, 20, 120]
    colors = ["red", "blue", "green", "yellow"]
    nchains = max(atom_data[:, 1])

    for row in atom_data:
        wrap_coords = wrap(row[-3:], Ls)
        # atom_id mol_id atom_type
        if style == "phase":
            ax.scatter(*wrap_coords, c=colors[int(row[1] <= nchains / 2.0)])
        else:
            ax.scatter(*wrap_coords, c=colors[int(row[2] - 1)])
    # ax.view_init(elev=20., azim=-35, roll=0)
    ax.view_init(elev=20.0, azim=-35)
    ax.set_axis_off()
    ax.set_box_aspect(aspect=(1, 1, 6))

    return


# dpath = '/Users/fanc/della-home/GS/polymer-test/03_04_pipeline_N3/N6-index303-mineig0.25/L30/diag-1000.0-9/coex-0.375/'
# import glob
# snaps = glob.glob(os.path.join(dpath, 'final*stitch-*.atom'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input .atom file")
    parser.add_argument("--phase", type=int, default=0, help="if plot mixing")
    parser.add_argument("--output", type=stry, help="output .svg file")

    clargs = parser.parse_args()

    fig = plt.figure(figsize=(60, 30))
    atom_data = np.array(read_atom_file(clargs.input))
    string = snaps[d].split("-")[-1].split(".")[0]
    a, b = string[0], string[1]
    print(a, b, order[string])
    plot_config(atom_data, fig, len(snaps), 1)
    plot_config(atom_data, fig, len(snaps), 1)

    if clargs.phase == 1:
        tot = 2
        plot_config(atom_data, fig, tot, 2, style="phase")

    fig.tight_layout()
