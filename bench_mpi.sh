#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --output=mpi_res.txt
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --time=00:20:00

module load gcc
module load openmpi

make

export OMP_NUM_THREADS=1
echo "Procs Time"
for P in 1 2 4 8 16; do
    T=$(mpirun -np $P ./poisson_solver | grep "Time:" | awk '{print $2}')
    echo "$P $T"
done
