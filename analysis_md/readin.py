import argparse, gzip, math
import numpy as np
from collections import defaultdict
import json, pickle
from contextlib import ExitStack

def wrap(pos, Ls):
    ans = []
    for i in range(len(pos)):
         ans.append(pos[i] - np.floor(pos[i] / Ls[i]) * Ls[i])
    return ans


def get_timestep_com(f):
    data = {}
    flag_time, flag_n, flag_box, pos = None, None, None, None
    atom_data = []
    box = []
    natoms = None
    for line in f:
        contents = line.split()
        if 'TIMESTEP' in contents:
            flag_time=True
            continue
        if 'NUMBER' in contents:
            flag_n = True
            continue
        if 'BOX' in contents:
            flag_box = True
            continue
        if 'mol' in contents and 'type' in contents:
            pos = True
            continue
        if flag_time:
            timestep = int(contents[0])
            flag_time = False
        if flag_n:
            natoms = int(contents[0])
            flag_n = False
        if flag_box and len(box) < 3:
            box.append([float(s) for s in contents])
        if pos:
            atom_data.append([float(s) for s in contents[:]])
        if natoms and len(atom_data) == natoms:
            # print("Read in timestep: ", timestep)
            break

    data['atom'] = np.array(atom_data)
    data['box'] = box
    data['timestep'] = timestep
    #data['nmols'] = nchains

    return data


def read_traj(traj_path):
    with open(traj_path, 'r') as com_file:
        while True:
            try:
                data = get_timestep_com(com_file)
                yield data
            except EOFError:
                break

#def read_traj(traj_path):
#    def get_timestep_com(f):
#        data = {}
#        flag_time, flag_n, flag_box, pos = None, None, None, None
#        atom_data = []
#        box = []
#        natoms = None
#        for line in f:
#            contents = line.split()
#            if 'TIMESTEP' in contents:
#                flag_time=True
#                continue
#            if 'NUMBER' in contents:
#                flag_n = True
#                continue
#            if 'BOX' in contents:
#                flag_box = True
#                continue
#            if 'mol' in contents and 'type' in contents:
#                pos = True
#                continue
#            if flag_time:
#                timestep = int(contents[0])
#                flag_time = False
#            if flag_n:
#                natoms = int(contents[0])
#                flag_n = False
#            if flag_box and len(box) < 3:
#                box.append([float(s) for s in contents])
#            if pos:
#                atom_data.append([float(s) for s in contents[:]])
#            if natoms and len(atom_data) == natoms:
#                # print("Read in timestep: ", timestep)
#                break
#        return timestep, atom_data
#
#    with open(traj_path, 'r') as com_file:
#        while True:
#            try:
#                timestep, data = get_timestep_com(com_file)
#                yield timestep, data
#            except EOFError:
#                break
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('compath', type=str, help="path to CoM file")

    clargs = parser.parse_args()
    dump_path = None
    counter = 0
    for frame in read_traj(clargs.compath):
        try:
            #timestep, data = lines
            print(frame.keys(), frame['timestep'])
            counter += 1
            if counter > 10:
                break
        except:
            print("break at ", counter)

#    with ExitStack() as stack:
#        foutput = stack.enter_context(gzip.open(clargs.output, 'wb'))
#        for lines in read_timesteps(dump_path, clargs.compath, clargs.gyrpath):
#            try:
#                timestep, molcom, molgyr = lines
#                counter += 1
#                #if counter > 3:
#                #    break
#
#                dr, rdf, n = calc_com_rdf(Ls, molcom, moltypes, ddr=0.5)
#                results = {'timestep': timestep, 'dr':dr, 'rdf': rdf, 'n' : n, 'moltypes': moltypes}
#                # with gzip.open("rdf-%d.p.gz"%counter, 'wb') as f:
#                pickle.dump(results, foutput)
#                foutput.flush()
#            except:
#                print("break at ", counter)
#                break
