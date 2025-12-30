cat > main.cpp << 'EOF'
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

const double A_PARAM = 1e5;
const double EPS = 1e-8;
const double X0 = -1.0, Y0 = -1.0, Z0 = -1.0;
const double DX = 2.0, DY = 2.0, DZ = 2.0;
const int Nx = 256; 
const int Ny = 256;
const int Nz = 256;

double phi_exact(double x, double y, double z) { return x*x + y*y + z*z; }
double rho_func(double x, double y, double z) { return 6.0 - A_PARAM * phi_exact(x, y, z); }

void calc_layer(int i, int start_i, int Nx_global, int Ny_local, int Nz_local, 
                size_t layer_size, double hx2, double hy2, double hz2, double denom,
                const std::vector<double>& phi, std::vector<double>& phi_new, 
                const std::vector<double>& rho_arr, double& max_diff) {
    
    int global_i = start_i + (i - 1);
    
    if (global_i > 0 && global_i < Nx_global - 1) {
        double local_diff = 0.0;
        
        #pragma omp parallel for collapse(2) reduction(max: local_diff)
        for(int j=1; j<Ny_local-1; ++j) {
            for(int k=1; k<Nz_local-1; ++k) {
                int idx = i*layer_size + j*Nz_local + k;
                
                double val = ((phi[idx+layer_size] + phi[idx-layer_size])/hx2 +
                              (phi[idx+Nz_local] + phi[idx-Nz_local])/hy2 +
                              (phi[idx+1] + phi[idx-1])/hz2 - rho_arr[idx]) / denom;
                
                phi_new[idx] = val;
                double diff = std::abs(val - phi[idx]);
                if (diff > local_diff) local_diff = diff;
            }
        }
        
        #pragma omp critical
        {
            if(local_diff > max_diff) max_diff = local_diff;
        }
    }
}

int main(int argc, char** argv) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double hx = DX / (Nx - 1), hy = DY / (Ny - 1), hz = DZ / (Nz - 1);
    double hx2 = hx*hx, hy2 = hy*hy, hz2 = hz*hz;
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
                rho_arr[idx] = rho_func(x, y, z);
                if (global_i == 0 || global_i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
                    phi[idx] = phi_exact(x, y, z);
                    phi_new[idx] = phi[idx];
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

        if (local_Nx > 2) {
            for (int i = 2; i < local_Nx; ++i) {
                calc_layer(i, start_i, Nx, Ny, Nz, layer_size, hx2, hy2, hz2, denom, phi, phi_new, rho_arr, max_diff);
            }
        }

        MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

        if (local_Nx >= 1) 
            calc_layer(1, start_i, Nx, Ny, Nz, layer_size, hx2, hy2, hz2, denom, phi, phi_new, rho_arr, max_diff);
        
        if (local_Nx > 1) 
            calc_layer(local_Nx, start_i, Nx, Ny, Nz, layer_size, hx2, hy2, hz2, denom, phi, phi_new, rho_arr, max_diff);

        std::swap(phi, phi_new);
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        max_diff = global_max_diff;

    } while (max_diff > EPS && it < 500);

    double end_time = MPI_Wtime();
    
    double local_err = 0.0;
    #pragma omp parallel for collapse(2) reduction(max: local_err)
    for(int i=1; i<=local_Nx; ++i) {
        for(int j=0; j<Ny; ++j) {
            for(int k=0; k<Nz; ++k) {
                int global_i = start_i + (i - 1);
                double x = X0 + global_i*hx, y = Y0 + j*hy, z = Z0 + k*hz;
                int idx = i*layer_size + j*Nz + k;
                double diff = std::abs(phi[idx] - phi_exact(x,y,z));
                if (diff > local_err) local_err = diff;
            }
        }
    }
    double global_err;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI=%d, OMP=%d, Time=%.4f, Error=%.2e\n", size, omp_get_max_threads(), end_time - start_time, global_err);
    }
    MPI_Finalize();
    return 0;
}
EOF
