import argparse, gzip, math
import numpy as np
from collections import defaultdict
from test import map_Mol_Sequence
import json, pickle
from contextlib import ExitStack
import canalyze


def wrap(pos, Ls):
    ans = []
    for i in range(len(pos)):
        ans.append(pos[i] - np.floor(pos[i] / Ls[i]) * Ls[i])
    return ans


def read_atom_file(filename):
    with open(filename, "r") as f:
        print("reading_atom_file input:", filename)
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
        return atom_data, box


def read_timesteps(dump_path, com_path, gyr_path):
    # Read a COM dump file and a Rg dump file simultaneously:
    def get_timestep_com(f):
        data = {}
        for line in f:
            if len(line) > 0 and line[0] != "#":
                if len(line.split()) == 2:
                    timestep, n = int(line.split()[0]), int(line.split()[1])
                else:
                    data[int(line.split()[0])] = tuple(
                        float(x) for x in line.split()[1:]
                    )
                    if len(data) == n:
                        break
        return timestep, data

    def get_timestep_gyr(f):
        data, n = {}, 0
        for line in f:
            if len(line) > 0 and line[0] != "#":
                if len(line.split()) == 2 and n == 0:
                    timestep, n = int(line.split()[0]), int(line.split()[1])
                else:
                    data[int(line.split()[0])] = float(line.split()[1])
                    if len(data) == n:
                        break
        return timestep, data

    with open(com_path, "r") as com_file, open(gyr_path, "r") as gyr_file:
        while True:
            try:
                timestep_molcom, molcom = get_timestep_com(com_file)
                timestep_molgyr, molgyr = get_timestep_gyr(gyr_file)
                if timestep_molcom == timestep_molgyr:
                    yield timestep_molcom, molcom, molgyr
                else:
                    raise Exception(
                        "timesteps do not match:", timestep_molcom, timestep_molgyr
                    )
            except EOFError:
                break


def calc_com_rdf(box, molcom, moltypes, ddr=1.0, maxdr=None, fast=True):
    # Compute the COM rdf using the Cython code in canalyze.pyx
    L = np.array([(box[d][1] - box[d][0]) for d in range(3)])
    if maxdr == None:
        maxdr = min(L) / 2.0
    return canalyze.calc_com_rdf_fast(L, ddr, maxdr, molcom, moltypes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("compath", type=str, help="path to CoM file")
    parser.add_argument("gyrpath", type=str, help="path to Rg file")
    parser.add_argument("input", type=str, help="path to phase description .json file")
    parser.add_argument("config", type=str, help="path to .atom file")
    parser.add_argument(
        "--output",
        type=str,
        default="rdf.p.gz",
        help="path to phase description .json file",
    )

    clargs = parser.parse_args()

    # Calculate radial distribution function for polymer COMs.
    # L: box dimensions
    # ddr: bin width for rdf
    # maxdf: cutoff for rdf (must be less than min(L)/2)
    # molcom: (N_molecules x 3) matrix of COM positions
    # moltypes: N_molecules-length dictionary assigning each molecule id 1,...,N_molecules to a molecule type

    # chain map
    with open(clargs.input, "r") as p:
        phases = json.load(p)

    atom_data, Ls = read_atom_file(clargs.config)
    print("box: ", Ls)
    # moltypes = {}
    moltypes = map_Mol_Sequence(atom_data, phases)
    # rdf = calc_com_rdf(Ls, molcom, moltypes)

    dump_path = None
    counter = 0

    with ExitStack() as stack:
        foutput = stack.enter_context(gzip.open(clargs.output, "wb"))
        for lines in read_timesteps(dump_path, clargs.compath, clargs.gyrpath):
            try:
                timestep, molcom, molgyr = lines
                counter += 1
                # if counter > 3:
                #    break

                dr, rdf, n = calc_com_rdf(Ls, molcom, moltypes, ddr=0.5)
                results = {
                    "timestep": timestep,
                    "dr": dr,
                    "rdf": rdf,
                    "n": n,
                    "moltypes": moltypes,
                }
                # with gzip.open("rdf-%d.p.gz"%counter, 'wb') as f:
                pickle.dump(results, foutput)
                foutput.flush()
            except:
                print("break at ", counter)
                break
