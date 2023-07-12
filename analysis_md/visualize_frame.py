import argparse
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations

from analysis_traj import ref_compositions
from analysis_lammps import read_atom_file, wrap
from analysis_atom import map_Mol_Sequence, bin_data

def plot_config(atom_data, box, ax, style=None, view='vertical'):
    Ls = [side[1]-side[0] for side in box]
       # FIXME double check the box origin is at (0,0,0)
    for s, e in combinations(np.array(list(product(box[0], box[1], box[2]))), 2):
        if np.sum(np.abs(s-e)) in Ls:
            ax.plot3D(*zip(s, e), color="black")
    colors = ['red', 'blue', 'green', 'yellow'] #FIXME handle the case with more than four monomer types
    nchains = max(atom_data[:, 1])
    for row in atom_data:
        wrap_coords = wrap(row[-3:], Ls)
        # atom_id mol_id atom_type
        if style == 'phase':
            ax.scatter(*wrap_coords, c=colors[int(row[1] <= nchains/2.0)])
        else:
            ax.scatter(*wrap_coords, c=colors[int(row[2]-1)])
    if view == 'vertical':
        ax.view_init(elev=20., azim=-35., roll=0.) # vertical view
    else:
        ax.view_init(elev=0., azim=45., roll=270.)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')

    #ax.set_box_aspect(aspect = (1,1,))#int(np.ceil(Ls[2]/Ls[1]))))
    ax.margins(x=0, y=0)
    return

def plot_profile(bins, profile, ax, ylabel=None, style=None):
    if style == 'chain':
        for i in range(profile.shape[1]):
            ax.plot(bins, profile[:, i], label='chain type %d' %i)
        ax.legend(loc='upper right', fontsize=12)
    elif style == 'op':
        for i in range(profile.shape[1]):
            ax.plot(bins, profile[:, i], label='phase-%d' %i)
        ax.legend(loc='upper right', fontsize=12)
    else:
        ax.plot(bins, profile)

    ax.set_xlabel('z', fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="input .atom file containing configuration")
    parser.add_argument('phase', type=str, help="input .json file containing phase information")
    parser.add_argument('--phase', type=int, default=0, help="if plot config color coded by initial phase label")
    parser.add_argument('--chain', type=int, default=0, help="if plot config color coded by chain type")
    parser.add_argument('--output', type=str, help="output .svg file")

    clargs = parser.parse_args()

    with open(clargs.phase, 'r') as p:
        phases = json.load(p)
    phis = []
    for i in phases['phases'].keys():
        phis.append(ref_compositions(phases['phases'][str(i)], phases))
    print(np.array(phis))

    results = read_atom_file(clargs.input)
    atom_data = results['atom']

    print("box dim: ", results['box'])

    Ls = [side[1]-side[0] for side in results['box']]
    string = clargs.input.split("/")[-1].split("-")[-1].split('.')[0]
    a, b = string[0], string[1]
    print('coexistence of %s-%s' %(a, b))

    tot = 1
    if clargs.phase == 1:
        tot += 1

    #fig = plt.figure(figsize=(15, 10))
    fig = plt.figure(figsize=plt.figaspect(1.))
    gs = fig.add_gridspec(nrows=3, ncols=3)
    #gs = gridspec.GridSpec(3, 3)
    ax_config = fig.add_subplot(gs[0, :], projection='3d')
    ax_chain = fig.add_subplot(gs[1, :])
    ax_op = fig.add_subplot(gs[2, :])

    plot_config(atom_data, results['box'], ax_config, view='horizontal')

    # bin .atom file based on chain types

    divider = sum(phases['phases'][str(int(a)+1)].values())
    print("divider: ", divider)
    bins = np.arange(0., Ls[-1], 3.)
    mol_sequence_map = map_Mol_Sequence(np.array(atom_data), phases['components'])
    profile = bin_data(atom_data, Ls, phis, mol_sequence_map, phases['N'], divider, bins)

    mid_bins = (bins[:-1] + bins[1:]) / 2.
    plot_profile(mid_bins, profile['chain_composition'], ax_chain, ylabel='composition', style='chain')
    plot_profile(mid_bins, profile['op'], ax_op, ylabel='order parameter', style='op')


    fig.tight_layout(pad=0.1)
    if clargs.output:
        fig.savefig(clargs.output, transparent=True)
