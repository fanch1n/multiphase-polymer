#!/bin/bash
#SBATCH --job-name=rdf           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
module purge
module load anaconda3/2022.10

scale=$1
dpath=/home/fanc/GS/polymer-test/03_04_pipeline_N3/N4-index4/comparison/mineig0.5/diag-100.0-7/NPT-production
srcpath=/home/fanc/lammps_analysis/src # src folder containing the analysis script, need to build before using
phase=$dpath/phase.json

for label in 0 1 2
do
    com=$dpath/rdf-$scale/phase-${label}_mol_com.dat
    gyr=$dpath/rdf-$scale/phase-${label}_mol_gyration.dat
    config=$dpath/rdf-$scale/end-phase-${label}.atom

    python3 $srcpath/analyze_lammps_data.py $com $gyr $phase $config --output=$dpath/rdf-$scale/rdf-phase-${label}.p.gz
done
