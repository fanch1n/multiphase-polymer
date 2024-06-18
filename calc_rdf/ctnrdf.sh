#!/bin/bash

scale=0.375 # This is the default scale used for generating the coeffs

folder="rdf-$scale"
echo $folder
cd $folder

for i in 0 1 2 # loop through phase label, change accordingly
do
    file=ctn-${i}.in
    simfile=ctn_test-${i}.sh

    # ctn.in is the input file, take a look at the calculations and corresponding output:
    # - computing the radius of gyration of each chain, output to _mol_gyration.dat
    # - computing the average radial distribution function, output to monomer_rdf.dat
    cp ../ctn.in $file
    cp ../run_test.sh $simfile

    str="phase-${i}"
    sed -i "s/REPLACE/${str}/g" $file
    sed -i -e "/PAIRCOEFF/r $pair_file" $file
    sed -i -e '/PAIRCOEFF/d' $file
    sed -i "s/REPLACE/${file}/g" $simfile

    sbatch slurm.sh $simfile
done

cd ..
