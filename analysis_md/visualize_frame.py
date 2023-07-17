import argparse
import numpy as np
import json

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations

from analysis_traj import ref_compositions
from analysis_lammps import read_atom_file, wrap, read_traj
from analysis_atom import map_Mol_Sequence, bin_data, get_shiftAtoms
from analysis_profile import find_condensed_phase_edges
from calc_fitness import try_fit, right_step, left_step, unitstep

def plot_config(atom_data, box, ax, style=None, view='vertical'):
    Ls = [side[1]-side[0] for side in box]
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
        # ax.view_init(elev=0., azim=45., roll=270.)
        ax.view_init(0., 45., 270.)
    ax.set_axis_off()
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_box_aspect(aspect = (1,1,int(np.ceil(Ls[2]/Ls[1]))))
    ax.margins(x=0, y=0)
    return

def plot_profile(bins, profile, ax, ylabel=None, style=None, errs=[], coex_labels=[]):
    for i in range(profile.shape[1]):
        if style == 'chain':
            ax.plot(bins, profile[:, i], label='chain-%d' %(1+i))
            if len(errs) > 0 and len(errs[:, i]) == len(profile[:, i]):
                ax.fill_between(bins, profile[:, i]-errs[:, i], profile[:, i]+errs[:, i], alpha=0.5)
        elif style == 'op':
            if i in coex_labels:
                ax.plot(bins, profile[:, i], label='phase-%d' %(1+i))
                if len(errs) > 0 and len(errs[:, i]) == len(profile[:, i]):
                    ax.fill_between(bins, profile[:, i]-errs[:, i], profile[:, i]+errs[:, i], alpha=0.5)
        else:
            ax.plot(bins, profile)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('z', fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="input .atom file containing a configuration")
    parser.add_argument('trajpath', type=str, help="input .lammpstrj file containing a list of configurations")
    parser.add_argument('phase', type=str, help="input .json file containing phase information")
    parser.add_argument('--plot-fitness', action='store_true', help="if plot the fitness template fitting[False]")
    parser.add_argument('--phase', type=int, default=0, help="if plot config color coded by initial phase label")
    parser.add_argument('--chain', type=int, default=0, help="if plot config color coded by chain type")
    parser.add_argument('--output', type=str, help="output .svg file")

    clargs = parser.parse_args()

    with open(clargs.phase, 'r') as p:
        phases = json.load(p)
    phis = []
    for i in phases['phases'].keys():
        if i != "0":
            phis.append(ref_compositions(phases['phases'][str(i)], phases))
    print(np.array(phis))

    with open(clargs.input, 'r') as file_in:
        results = read_atom_file(file_in)

    atom_data = results['atom']
    print("box dim: ", results['box'])
    Ls = [side[1]-side[0] for side in results['box']]
    #FIXME check the origin is at (0, 0, 0)

    string = clargs.input.split("/")[-1].split("-")[-1].split('.')[0]
    alpha, beta = int(string[0]), int(string[1])
    print('coexistence of %d-%d' %(alpha, beta))

    #fig0 = plt.figure(figsize=(3, 12))
    #fig0=plt.figure()
    #gs0 = fig0.add_gridspec(nrows=1, ncols=3)
    #ax_config = fig0.add_subplot(gs0[0, :], projection='3d')

    fig = plt.figure(figsize=(6, 12))
    fig = plt.figure(figsize=plt.figaspect(1.))
    gs = fig.add_gridspec(nrows=2, ncols=3)
    #ax_config = fig.add_subplot(gs[0, :], projection='3d')
    ax_chain = fig.add_subplot(gs[0, :])
    ax_op = fig.add_subplot(gs[1, :])

    divider = sum(phases['phases'][str(int(alpha))].values())
    print("divider: ", divider)
    bins = np.arange(0., Ls[-1], 3.)
    mol_sequence_map = map_Mol_Sequence(np.array(atom_data), phases['components'])
    # currently the interface is determined based on binned profile instead of raw atom data
    def shift_center(atom_data, Ls, phis, mol_sequence_map, ntypes, divider):
        profile = bin_data(atom_data, Ls, phis, mol_sequence_map, ntypes, divider, bins)
        left_edge, right_edge = find_condensed_phase_edges(profile['density'], bins)
        dz = Ls[-1]/2. - (right_edge + left_edge)/2.
        return get_shiftAtoms(atom_data, np.array([0.,0.,dz]), Ls)

    shifted_atom_data = shift_center(atom_data, Ls, phis, mol_sequence_map, phases['N'], divider)

    #plot_config(shifted_atom_data, results['box'], ax_config, view='horizontal')
    #profile = bin_data(atom_data, Ls, phis, mol_sequence_map, phases['N'], divider, bins)

    mid_bins = (bins[:-1] + bins[1:]) / 2.
    #plot_profile(mid_bins, profile['op'], ax_op, ylabel='order parameter', style='op', coex_labels=[alpha-1, beta-1])
    #plot_profile(mid_bins, profile['chain_composition'], ax_chain, ylabel='composition', style='chain')
    list_compos = []
    list_ops = []
    freq = 500
    counter = 0
    for frame in read_traj(clargs.trajpath):
        try:
            counter += 1
            if counter % freq == 0:
                shifted_atoms = shift_center(frame['atom'], Ls, phis, mol_sequence_map, phases['N'], divider)
                profile = bin_data(shifted_atoms, Ls, phis, mol_sequence_map, phases['N'], divider, bins)
                list_compos.append(profile['chain_composition'])
                list_ops.append(profile['op'])
                print(counter)
        except:
            print("break at ", counter)
            break

    plot_profile(mid_bins, np.average(np.array(list_compos), axis=0), ax_chain, \
            ylabel='composition', errs=np.std(np.array(list_compos), axis=0), style='chain')

    avg_ops = np.average(np.array(list_ops), axis=0)
    plot_profile(mid_bins, avg_ops , ax_op, ylabel='order parameter',\
            style='op', errs=np.std(np.array(list_ops), axis=0), coex_labels=[alpha-1, beta-1])

    shared = np.dot(phis[alpha-1], phis[beta-1])
    # assuming alpha phase is on the left side of the box
    if clargs.plot_fitness:
        #FIXME scale parameter should be set properly
        if shared > 1e-3:
            left_func = left_step
            right_func  = right_step
        else:
            left_func = right_func = unitstep
        xData = np.array(mid_bins)
        left_yData = avg_ops[:, alpha-1]
        left_fittedParameters, Rsquared, left_RMSE = try_fit(xData, left_yData, left_func)
        left_modelPredictions = left_func(xData, *left_fittedParameters)
        ax_op.plot(xData, left_modelPredictions, label='fitness = %.2f' %(left_RMSE))

        right_yData = avg_ops[:, beta-1]
        right_fittedParameters, Rsquared, right_RMSE = try_fit(xData, right_yData, right_func)
        right_modelPredictions = right_func(xData, *right_fittedParameters)
        ax_op.plot(xData, right_modelPredictions, label='fitness = %.2f' %(right_RMSE))
        ax_op.legend(loc='lower center', fontsize=9)

    fig.tight_layout(pad=0.1)
    if clargs.output:
        if clargs.plot_fitness:
            fig.savefig(clargs.output+'-rmse-%.3f-%.3f.svg'%(left_RMSE, right_RMSE), transparent=True)
        else:
            fig.savefig(clargs.output+'.svg', transparent=True)
        #fig0.savefig(clargs.output+'-config.svg', transparent=True)
