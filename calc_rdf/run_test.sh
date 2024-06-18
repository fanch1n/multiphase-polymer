#!/bin/bash
set -euf -o pipefail
readonly gpu_count=${1:-$(nvidia-smi --list-gpus | wc -l)}

echo "Running Lennard Jones 8x4x8 example on ${gpu_count} GPUS..."
mpirun -n ${gpu_count} lmp -k on g ${gpu_count} -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -in REPLACE
