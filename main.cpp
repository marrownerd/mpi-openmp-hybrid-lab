/*
 * Лабораторная работа №4: Параллельная реализация метода Якоби
 * Уравнение: Delta phi - a*phi = rho
 * Область: [-1, 1] x [-1, 1] x [-1, 1]
 * Технологии: Hybrid MPI + OpenMP
 */

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


const double A_PARAM = 1e5;
const double EPS = 1e-8;
const double X0 = -1.0, Y0 = -1.0, Z0 = -1.0;
const double DX = 2.0, DY = 2.0, DZ = 2.0;


const int Nx = 256;
const int Ny = 256;
const int Nz = 256;


double phi_exact(double x, double y, double z) {
    return x*x + y*y + z*z;
}


double rho_func(double x, double y, double z) {
    return 6.0 - A_PARAM * phi_exact(x, y, z);
}

int main(int argc, char** argv) {
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double hx = DX / (Nx - 1);
    double hy = DY / (Ny - 1);
    double hz = DZ / (Nz - 1);

    double hx2 = hx * hx;
    double hy2 = hy * hy;
    double hz2 = hz * hz;
    double denom = 2.0/hx2 + 2.0/hy2 + 2.0/hz2 + A_PARAM;

    int local_Nx = Nx / size;
    int remainder = Nx % size;
    int start_i = rank * local_Nx + std::min(rank, remainder);
    if (rank < remainder) local_Nx++;
    
    size_t layer_size = Ny * Nz;
    size_t total_size = (local_Nx + 2) * layer_size;
    
    std::vector<double> phi(total_size, 0.0);
    std::vector<double> phi_new(total_size, 0.0);
    std::vector<double> rho_arr(total_size, 0.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < local_Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                int global_i = start_i + (i - 1); 
                double x = X0 + global_i * hx;
                double y = Y0 + j * hy;
                double z = Z0 + k * hz;
                
                int idx = i * layer_size + j * Nz + k;

                // Заполняем правую часть
                rho_arr[idx] = rho_func(x, y, z);

                if (global_i == 0 || global_i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
                    phi[idx] = phi_exact(x, y, z);
                    phi_new[idx] = phi[idx]; 
                } else {
                    phi[idx] = 0.0; 
                }
            }
        }
    }

    double max_diff = 0.0;
    int it = 0;
    double start_time = MPI_Wtime();

    do {
        it++;
        max_diff = 0.0;

        MPI_Request reqs[4];
        int n_reqs = 0;

        if (rank > 0) {
            MPI_Isend(&phi[1 * layer_size], layer_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(&phi[0 * layer_size], layer_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        if (rank < size - 1) {
            MPI_Isend(&phi[local_Nx * layer_size], layer_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(&phi[(local_Nx + 1) * layer_size], layer_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // граничные слои локальной области 
        auto calculate_layer = [&](int i) {
            int global_i = start_i + (i - 1);
            if (global_i > 0 && global_i < Nx - 1) { // НЕ пересчитываем глобальные границы
                double diff_local = 0.0;
                #pragma omp parallel for collapse(2) reduction(max: diff_local)
                for (int j = 1; j < Ny - 1; ++j) {
                    for (int k = 1; k < Nz - 1; ++k) {
                        int idx = i * layer_size + j * Nz + k;
                        double term_x = (phi[idx + layer_size] + phi[idx - layer_size]) / hx2;
                        double term_y = (phi[idx + Nz] + phi[idx - Nz]) / hy2;
                        double term_z = (phi[idx + 1] + phi[idx - 1]) / hz2;
                        
                        double val = (term_x + term_y + term_z - rho_arr[idx]) / denom;
                        phi_new[idx] = val;
                        diff_local = std::max(diff_local, std::abs(val - phi[idx]));
                    }
                }
                #pragma omp atomic update // Для обновления глобального максимума безопасно
                if (diff_local > max_diff) max_diff = diff_local; // учше через critical?
            }
        };

        // вычисление внутренней части (i = 2 ... local_Nx - 1)
        double local_max_diff = 0.0;
        if (local_Nx > 2) {
            #pragma omp parallel for collapse(2) reduction(max: local_max_diff)
            for (int i = 2; i < local_Nx; ++i) {
                // ... (то же самое, что внутри calculate_layer, дублируем для инлайнинга OpenMP)
                 for (int j = 1; j < Ny - 1; ++j) {
                    for (int k = 1; k < Nz - 1; ++k) {
                        int idx = i * layer_size + j * Nz + k;
                        double term_x = (phi[idx + layer_size] + phi[idx - layer_size]) / hx2;
                        double term_y = (phi[idx + Nz] + phi[idx - Nz]) / hy2;
                        double term_z = (phi[idx + 1] + phi[idx - 1]) / hz2;
                        
                        double val = (term_x + term_y + term_z - rho_arr[idx]) / denom;
                        phi_new[idx] = val;
                        local_max_diff = std::max(local_max_diff, std::abs(val - phi[idx]));
                    }
                }
            }
        }
        if (local_max_diff > max_diff) max_diff = local_max_diff;

        // ожидание завершения обменов
        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

        // считаем границы (i=1 и i=local_Nx), так как ghost cells обновлены
        // в классической схеме перекрытия мы считаем ВНУТРЕННОСТЬ пока идут обмены
        // ран зависящие от ghost cells считаем ПОСЛЕ Waitall.
        // в методе Якоби i=1 зависит от i=0(ghost) и i=2(своего).
        // поэтому i=1 и i=local_Nx считаем ЗДЕСЬ после Waitall.
        
        double border_max_diff = 0.0;
        // i = 1
        if (local_Nx >= 1) {
             // копи логику первог слоя
             int i = 1;
             int global_i = start_i + (i - 1);
             if (global_i > 0 && global_i < Nx - 1) {
                 #pragma omp parallel for collapse(2) reduction(max: border_max_diff)
                 for(int j=1; j<Ny-1; ++j) {
                    for(int k=1; k<Nz-1; ++k) {
                        int idx = i * layer_size + j * Nz + k;
                        double val = ((phi[idx+layer_size] + phi[idx-layer_size])/hx2 + 
                                      (phi[idx+Nz] + phi[idx-Nz])/hy2 + 
                                      (phi[idx+1] + phi[idx-1])/hz2 - rho_arr[idx]) / denom;
                        phi_new[idx] = val;
                        border_max_diff = std::max(border_max_diff, std::abs(val - phi[idx]));
                    }
                 }
             }
        }
        // i = local_Nx (если он отличается от 1)
        if (local_Nx > 1) {
             int i = local_Nx;
             int global_i = start_i + (i - 1);
             if (global_i > 0 && global_i < Nx - 1) {
                 #pragma omp parallel for collapse(2) reduction(max: border_max_diff)
                 for(int j=1; j<Ny-1; ++j) {
                    for(int k=1; k<Nz-1; ++k) {
                        int idx = i * layer_size + j * Nz + k;
                        double val = ((phi[idx+layer_size] + phi[idx-layer_size])/hx2 + 
                                      (phi[idx+Nz] + phi[idx-Nz])/hy2 + 
                                      (phi[idx+1] + phi[idx-1])/hz2 - rho_arr[idx]) / denom;
                        phi_new[idx] = val;
                        border_max_diff = std::max(border_max_diff, std::abs(val - phi[idx]));
                    }
                 }
             }
        }
        if (border_max_diff > max_diff) max_diff = border_max_diff;

        std::swap(phi, phi_new);

        // Сбор глобальной ошибки
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        max_diff = global_max_diff;

    } while (max_diff > EPS);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Converged in %d iterations.\n", it);
        printf("Time: %f seconds.\n", end_time - start_time);
        printf("Grid: %dx%dx%d\n", Nx, Ny, Nz);
        printf("Processes: %d\n", size);
    }

    MPI_Finalize();
    return 0;
}