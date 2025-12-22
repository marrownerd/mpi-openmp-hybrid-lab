#!/bin/bash
#SBATCH --job-name=hybrid_test
#SBATCH --output=hybrid_res.txt
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --time=00:20:00

module load gcc
module load openmpi

make

echo "Conf Time"
export OMP_NUM_THREADS=16
T=$(srun --ntasks=4 --cpus-per-task=16 ./poisson_solver | grep "Time:" | awk '{print $2}')
echo "4x16 $T"

export OMP_NUM_THREADS=8
T=$(srun --ntasks=8 --cpus-per-task=8 ./poisson_solver | grep "Time:" | awk '{print $2}')
echo "8x8 $T"

export OMP_NUM_THREADS=4
T=$(srun --ntasks=16 --cpus-per-task=4 ./poisson_solver | grep "Time:" | awk '{print $2}')
echo "16x4 $T"

export OMP_NUM_THREADS=1
T=$(srun --ntasks=64 --cpus-per-task=1 ./poisson_solver | grep "Time:" | awk '{print $2}')
echo "64x1 $T"
