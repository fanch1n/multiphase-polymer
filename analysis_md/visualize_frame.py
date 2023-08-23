import argparse
import numpy as np
import json
from itertools import product, combinations

from analysis_traj import ref_compositions
from analysis_atom import *
from analysis_profile import find_condensed_phase_edges

from calc_fitness import try_fit, right_step, left_step, unitstep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


def plot_config(atom_data, box, ax, style=None, view="vertical", style_map={}):
    Ls = [side[1] - side[0] for side in box]
    print("plotting MD snapshot with box dim: \n", box)
    for s, e in combinations(np.array(list(product(box[0], box[1], box[2]))), 2):
        if np.sum(np.abs(s - e)) in Ls:
            ax.plot3D(*zip(s, e), color="black")
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "aqua",
        "cyan",
    ]  # FIXME handle the case with more types
    nchains = max(atom_data[:, 1])
    for row in atom_data:
        # row: atom_id mol_id atom_type
        wrap_coords = wrap(row[-3:], Ls)
        if style == "phase":
            if len(style_map) > 0:
                # style map is the phase_map
                ax.scatter(*wrap_coords, c=colors[int(style_map[str(int(row[1]))])])
        elif style == "chain":
            # style_map is the mol_sequence_map
            if len(style_map) > 0:
                ax.scatter(*wrap_coords, c=colors[int(style_map[str(int(row[1]))])])
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


def plot_profile(
    bins,
    profile,
    ax,
    ylabel=None,
    style=None,
    lower_errs=[],
    upper_errs=[],
    coex_labels=[],
):
    pcolor = ["blue", "orange", "green", "cyan"]
    cflag = 0
    for i in range(profile.shape[1]):
        if style == "chain":
            ax.plot(bins, profile[:, i], label="chain-%d" % (1 + i))
            if len(lower_errs) > 0 and len(upper_errs) > 0:
                ax.fill_between(
                    bins,
                    lower_errs[:, i],
                    upper_errs[:, i],
                    alpha=0.5,
                )

        elif style == "op":
            if i in coex_labels:
                ax.plot(
                    bins, profile[:, i], label="phase-%d" % (1 + i), color=pcolor[cflag]
                )
                cflag += 1
                if len(lower_errs) > 0 and len(upper_errs) > 0:
                    ax.fill_between(
                        bins,
                        lower_errs[:, i],
                        upper_errs[:, i],
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
        "--plot-fitness",
        action="store_true",
        help="if plot the fitness template fitting[False]",
    )

    parser.add_argument(
        "--plot-phase",
        action="store_true",
        help="plot config color coded by initial phase label[False]",
    )
    parser.add_argument(
        "--plot-chain",
        action="store_true",
        help="plot config color coded by chain types[False]",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the processed profile [False]",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="frequency for processing the lammpstrj file",
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

    # with open(clargs.input, "r") as file_in:
    #    results = read_atom_file(file_in)

    # FIXME check the origin is at (0, 0, 0)
    string = clargs.input.split("/")[-1].split("-")[-1].split(".")[0]
    alpha, beta = int(string[0]), int(string[1])
    print("coexistence of %d-%d" % (alpha, beta))

    # bins = np.arange(0.0, Ls[-1], 3.0)
    # mid_bins = (bins[:-1] + bins[1:]) / 2.0
    # mol_sequence_map = map_Mol_Sequence(np.array(atom_data), phases["components"])
    # currently the interface is determined based on binned profile instead of raw atom data

    list_compos = []
    list_ops = []
    counter = 0

    read_func = None
    if ".gz" in clargs.input:
        read_func = read_traj_zipped
    else:
        read_func = read_traj

    for frame in read_func(clargs.input):
        try:
            if counter % clargs.freq == 0:
                if counter == 0:
                    atom_data = frame["atom"]
                    Ls = [side[1] - side[0] for side in frame["box"]]
                    box = frame["box"]
                    bins = np.arange(0.0, Ls[-1], 3.0)
                    mid_bins = (bins[:-1] + bins[1:]) / 2.0
                    mol_sequence_map = map_Mol_Sequence(
                        np.array(atom_data), phases["components"]
                    )

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
                print(counter, flush=True)

            counter += 1
        except:
            print("break at ", counter)
            break

    # plotting snapshot
    fig0 = plt.figure()
    ax_config = fig0.add_subplot(111, projection="3d")
    plot_config(shifted_atoms, box, ax_config, view="horizontal")
    out_config = clargs.output + "-config.png"
    fig0.savefig(out_config, dpi=1200, transparent=True)

    if clargs.plot_phase:
        fig1 = plt.figure()
        ax_label = fig1.add_subplot(111, projection="3d")
        plot_config(
            shifted_atoms,
            box,
            ax_label,
            style="phase",
            view="horizontal",
            style_map=phase_label_map,
        )
        fig1.savefig(clargs.output + "-phase.png", dpi=1200, transparent=True)

    # plotting avergage density profile and other parameter profile
    fig = plt.figure(figsize=plt.figaspect(1.5))
    gs = fig.add_gridspec(nrows=8, ncols=3, wspace=0, hspace=0)
    ax_insert = fig.add_subplot(gs[0:2, :])
    ax_chain = fig.add_subplot(gs[2:5, :])
    ax_op = fig.add_subplot(gs[5:, :])

    if clargs.save:
        np.save("proc-%d%d-compo.npy" % (alpha, beta), np.array(list_compos))
        np.save("proc-%d%d-ops.npy" % (alpha, beta), np.array(list_ops))

    plot_profile(
        mid_bins,
        np.average(np.array(list_compos), axis=0),
        ax_chain,
        ylabel="composition",
        lower_errs=np.quantile(np.array(list_compos), 0.25, axis=0),
        upper_errs=np.quantile(np.array(list_compos), 0.75, axis=0),
        style="chain",
    )

    avg_ops = np.average(np.array(list_ops), axis=0)
    plot_profile(
        mid_bins,
        avg_ops,
        ax_op,
        ylabel="order parameter",
        style="op",
        lower_errs=np.quantile(np.array(list_ops), 0.25, axis=0),
        upper_errs=np.quantile(np.array(list_ops), 0.75, axis=0),
        coex_labels=[alpha - 1, beta - 1],
    )

    shared = (
        np.dot(phis[alpha - 1], phis[beta - 1])
        / np.linalg.norm(phis[alpha - 1], ord=2)
        / np.linalg.norm(phis[beta - 1], ord=2)
    )

    # alpha phase is assumed to be on the left side of the box
    if clargs.plot_fitness:
        if 1e-3 < shared < 0.99:
            left_func = left_step
            right_func = right_step
        else:
            left_func = right_func = unitstep
        xData = np.array(mid_bins)
        left_yData = avg_ops[:, alpha - 1]
        left_fittedParameters, Rsquared, left_MSE = try_fit(
            xData, left_yData, left_func
        )
        left_modelPredictions = left_func(xData, *left_fittedParameters)
        ax_op.plot(
            xData,
            left_modelPredictions,
            label="MSD = %.2f" % (left_MSE),
            color="blue",
            linestyle="dashed",
        )

        right_yData = avg_ops[:, beta - 1]
        right_fittedParameters, Rsquared, right_MSE = try_fit(
            xData, right_yData, right_func
        )
        right_modelPredictions = right_func(xData, *right_fittedParameters)
        ax_op.plot(
            xData,
            right_modelPredictions,
            label="MSD = %.2f" % (right_MSE),
            color="orange",
            linestyle="dashed",
        )
        ax_op.legend(loc="lower center", fontsize=9)

    out_profile = clargs.output + ".svg"
    config = trim(Image.open(out_config))
    im = ax_insert.imshow(config)
    ax_insert.set_axis_off()
    if clargs.output:
        fig.tight_layout(pad=0.1)
        fig.savefig(out_profile)
