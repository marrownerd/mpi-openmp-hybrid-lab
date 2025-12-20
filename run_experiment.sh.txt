#!/bin/bash

make clean
make

echo "Running experiments on 4 nodes..."
echo "Proc | Threads | Time" > results.csv

# Варианты конфигураций для 4 узлов и 64 ядер суммарно:
# 1. 4 MPI процесса (по 1 на узел), 16 потоков на процесс
# 2. 8 MPI процессов (по 2 на узел), 8 потоков на процесс
# 3. 16 MPI процессов (по 4 на узел), 4 потока на процесс
# 4. 32 MPI процесса (по 8 на узел), 2 потока на процесс
# 5. 64 MPI процесса (по 16 на узел), 1 поток (чистый MPI)

declare -a configs=(
    "4 16"
    "8 8"
    "16 4"
    "32 2"
    "64 1"
)

for config in "${configs[@]}"; do
    set -- $config
    MPI_PROCS=$1
    OMP_THREADS=$2
    
    # Установка переменных окружения для OpenMP
    export OMP_NUM_THREADS=$OMP_THREADS
    
    # Привязка потоков к ядрам важна для производительности
    export OMP_PROC_BIND=true 
    export OMP_PLACES=cores

    echo "Testing: $MPI_PROCS MPI processes x $OMP_THREADS OpenMP threads"
    
    # Запуск. Опция --map-by node:PE=$OMP_THREADS нужна для OpenMPI, 
    # чтобы правильно раскидать процессы по узлам и зарезервировать ядра под потоки.
    # Если используется Slurm: srun -n $MPI_PROCS -c $OMP_THREADS ./poisson_solver
    
    # Пример для OpenMPI:
    # mpirun -np $MPI_PROCS --map-by ppr:$((MPI_PROCS/4)):node:pe=$OMP_THREADS ./poisson_solver > temp_output.txt
    
    # Для локального теста (если у вас нет кластера, а просто мощная машина):
    mpirun -np $MPI_PROCS ./poisson_solver > temp_output.txt
    
    TIME=$(grep "Time:" temp_output.txt | awk '{print $2}')
    echo "$MPI_PROCS | $OMP_THREADS | $TIME" >> results.csv
    
    echo "Done. Time: $TIME s"
done

echo "Best configuration found:"
sort -t "|" -k 3n results.csv | head -n 1