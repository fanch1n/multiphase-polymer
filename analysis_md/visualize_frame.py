import argparse
import numpy as np
import json
from itertools import product, combinations
from analysis_traj import ref_compositions
from analysis_lammps import read_atom_file, wrap, read_traj
from analysis_atom import map_Mol_Sequence, bin_data, get_shiftAtoms
from analysis_profile import find_condensed_phase_edges
from calc_fitness import try_fit, right_step, left_step, unitstep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


def plot_config(atom_data, box, ax, style=None, view="vertical"):
    Ls = [side[1] - side[0] for side in box]
    for s, e in combinations(np.array(list(product(box[0], box[1], box[2]))), 2):
        if np.sum(np.abs(s - e)) in Ls:
            ax.plot3D(*zip(s, e), color="black")
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
    ]  # FIXME handle the case with more than four monomer types
    nchains = max(atom_data[:, 1])
    for row in atom_data:
        wrap_coords = wrap(row[-3:], Ls)
        # atom_id mol_id atom_type
        if style == "phase":
            ax.scatter(*wrap_coords, c=colors[int(row[1] <= nchains / 2.0)])
        else:
            ax.scatter(*wrap_coords, c=colors[int(row[2] - 1)])
    if view == "vertical":
        ax.view_init(elev=20.0, azim=-35.0, roll=0.0)  # vertical view
    else:
        ax.view_init(0.0, 45.0, 270.0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(x=0, y=0)
    return


def plot_profile(bins, profile, ax, ylabel=None, style=None, errs=[], coex_labels=[]):
    for i in range(profile.shape[1]):
        if style == "chain":
            ax.plot(bins, profile[:, i], label="chain-%d" % (1 + i))
            if len(errs) > 0 and len(errs[:, i]) == len(profile[:, i]):
                ax.fill_between(
                    bins,
                    profile[:, i] - errs[:, i],
                    profile[:, i] + errs[:, i],
                    alpha=0.5,
                )
        elif style == "op":
            if i in coex_labels:
                ax.plot(bins, profile[:, i], label="phase-%d" % (1 + i))
                if len(errs) > 0 and len(errs[:, i]) == len(profile[:, i]):
                    ax.fill_between(
                        bins,
                        profile[:, i] - errs[:, i],
                        profile[:, i] + errs[:, i],
                        alpha=0.5,
                    )
        else:
            ax.plot(bins, profile)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("z", fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    return


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        # Failed to find the borders, convert to "RGB"
        return trim(im.convert("RGB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=str, help="input .atom file containing a configuration"
    )
    parser.add_argument(
        "trajpath",
        type=str,
        help="input .lammpstrj file containing a list of configurations",
    )
    parser.add_argument(
        "phase", type=str, help="input .json file containing phase information"
    )
    parser.add_argument(
        "--plot-fitness",
        action="store_true",
        help="if plot the fitness template fitting[False]",
    )
    parser.add_argument(
        "--summary", action="store_true", help="if make the summary plot [False]"
    )
    parser.add_argument(
        "--plot-phase",
        action="store_true",
        help="plot config color coded by initial phase label",
    )
    parser.add_argument(
        "--chain", type=int, default=0, help="if plot config color coded by chain type"
    )
    parser.add_argument("--output", type=str, help="output .svg file")

    clargs = parser.parse_args()

    with open(clargs.phase, "r") as p:
        phases = json.load(p)
    phis = []
    for i in phases["phases"].keys():
        if i != "0":
            phis.append(ref_compositions(phases["phases"][str(i)], phases))
    print(np.array(phis))

    with open(clargs.input, "r") as file_in:
        results = read_atom_file(file_in)

    atom_data = results["atom"]
    print("box dim: ", results["box"])
    Ls = [side[1] - side[0] for side in results["box"]]
    # FIXME check the origin is at (0, 0, 0)

    string = clargs.input.split("/")[-1].split("-")[-1].split(".")[0]
    alpha, beta = int(string[0]), int(string[1])
    print("coexistence of %d-%d" % (alpha, beta))

    fig0 = plt.figure()
    ax_config = fig0.add_subplot(111, projection="3d")

    fig = plt.figure(figsize=plt.figaspect(1.0))
    gs = fig.add_gridspec(nrows=2, ncols=3)
    ax_chain = fig.add_subplot(gs[0, :])
    ax_op = fig.add_subplot(gs[1, :])

    divider = sum(phases["phases"][str(int(alpha))].values())
    print("divider: ", divider)

    bins = np.arange(0.0, Ls[-1], 3.0)
    mid_bins = (bins[:-1] + bins[1:]) / 2.0

    mol_sequence_map = map_Mol_Sequence(np.array(atom_data), phases["components"])
    # currently the interface is determined based on binned profile instead of raw atom data
    def shift_center(atom_data, Ls, phis, mol_sequence_map, ntypes, divider):
        profile = bin_data(atom_data, Ls, phis, mol_sequence_map, ntypes, divider, bins)
        left_edge, right_edge = find_condensed_phase_edges(profile["density"], bins)
        dz = Ls[-1] / 2.0 - (right_edge + left_edge) / 2.0
        return get_shiftAtoms(atom_data, np.array([0.0, 0.0, dz]), Ls)

    # plot the snapshot
    shifted_atom_data = shift_center(
        atom_data, Ls, phis, mol_sequence_map, phases["N"], divider
    )
    plot_config(shifted_atom_data, results["box"], ax_config, view="horizontal")

    list_compos = []
    list_ops = []
    freq = 500
    counter = 0
    for frame in read_traj(clargs.trajpath):
        try:
            counter += 1
            if counter % freq == 0:
                shifted_atoms = shift_center(
                    frame["atom"], Ls, phis, mol_sequence_map, phases["N"], divider
                )
                profile = bin_data(
                    shifted_atoms,
                    Ls,
                    phis,
                    mol_sequence_map,
                    phases["N"],
                    divider,
                    bins,
                )
                list_compos.append(profile["chain_composition"])
                list_ops.append(profile["op"])
                print(counter)
        except:
            print("break at ", counter)
            break

    # plotting avergage density profile and other parameter profile
    plot_profile(
        mid_bins,
        np.average(np.array(list_compos), axis=0),
        ax_chain,
        ylabel="composition",
        errs=np.std(np.array(list_compos), axis=0),
        style="chain",
    )

    avg_ops = np.average(np.array(list_ops), axis=0)
    plot_profile(
        mid_bins,
        avg_ops,
        ax_op,
        ylabel="order parameter",
        style="op",
        errs=np.std(np.array(list_ops), axis=0),
        coex_labels=[alpha - 1, beta - 1],
    )

    shared = np.dot(phis[alpha - 1], phis[beta - 1])
    # alpha phase is assumed to be on the left side of the box
    if clargs.plot_fitness:  # FIXME scale parameter should be set properly
        if 1e-3 < shared < 0.99:
            left_func = left_step
            right_func = right_step
        else:
            left_func = right_func = unitstep
        xData = np.array(mid_bins)
        left_yData = avg_ops[:, alpha - 1]
        left_fittedParameters, Rsquared, left_RMSE = try_fit(
            xData, left_yData, left_func
        )
        left_modelPredictions = left_func(xData, *left_fittedParameters)
        ax_op.plot(xData, left_modelPredictions, label="fitness = %.2f" % (left_RMSE))

        right_yData = avg_ops[:, beta - 1]
        right_fittedParameters, Rsquared, right_RMSE = try_fit(
            xData, right_yData, right_func
        )
        right_modelPredictions = right_func(xData, *right_fittedParameters)
        ax_op.plot(xData, right_modelPredictions, label="fitness = %.2f" % (right_RMSE))
        ax_op.legend(loc="lower center", fontsize=9)

    fig.tight_layout(pad=0.1)

    out_profile = clargs.output + ".png"
    out_config = clargs.output + "-config.png"
    if clargs.output:
        if clargs.plot_fitness:
            out_config = clargs.output + "-rmse-%.3f-%.3f.png" % (left_RMSE, right_RMSE)
        fig.savefig(out_profile, transparent=True)
        fig0.savefig(out_config, transparent=True)

    ### combine plots to the make the summary plots
    if clargs.summary:
        config = trim(Image.open(out_config))
        L = 400
        profile = Image.open(out_profile)
        profile = profile.resize((L, L))

        fig_ = plt.figure()
        gss = fig_.add_gridspec(nrows=5, ncols=1, wspace=0, hspace=0)
        ax1 = fig_.add_subplot(gss[0, :])
        ax2 = fig_.add_subplot(gss[1:, :])
        im = ax1.imshow(config)
        im2 = ax2.imshow(profile)

        ax1.set_axis_off()
        ax2.set_axis_off()
        fig_.tight_layout()
        fig_.savefig(clargs.output + "summary.png")
