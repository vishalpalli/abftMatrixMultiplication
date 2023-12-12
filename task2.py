'''
NAME: VISHAL PALLI
ID: 49042025
COURSE => ECE8379: FAULT TOLERANT COMPUTING

'''
from mpi4py import MPI
import numpy as np
import time
import os

def matrix_multiply(A, B):
    return np.dot(A, B)

def inject_error(matrix, error_rate):
    # Simulate injecting errors into the matrix
    errors = np.random.rand(*matrix.shape) < error_rate
    matrix[errors] = np.random.rand(*matrix.shape)[errors]

def check_errors(A, B, C):
    # Check for errors using ABFT
    AB = np.dot(A, B)
    error_matrix = AB - C
    error_detected = not np.allclose(error_matrix, np.zeros_like(error_matrix))
    return error_detected

def distribute_matrices(A, B, comm):
    # Get MPI rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Matrix dimensions
    m, n, p = A.shape[0], A.shape[1], B.shape[1]

    # Divide matrices among processes
    local_m = m // size
    local_A = np.zeros((local_m, n))
    local_B = np.zeros((n, p))

    comm.Scatter(A, local_A, root=0)
    comm.Bcast(B, root=0)

    return local_A, B

def gather_results(C_local, comm):
    # Gather local results to rank 0
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        result_matrix = np.empty_like(C_local)
    else:
        result_matrix = None

    comm.Gather(C_local, result_matrix, root=0)

    return result_matrix

def write_matrix_to_csv(matrix, file_path):
    # Write matrix to CSV file
    np.savetxt(file_path, matrix, delimiter=',')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Read matrices A and B from CSV files
        file_path_A = "matrix1.csv"
        file_path_B = "matrix2.csv"

        A = np.genfromtxt(file_path_A, delimiter=',')
        B = np.genfromtxt(file_path_B, delimiter=',')

        # Matrix multiplication without errors
        start_time = time.time()
        C = matrix_multiply(A, B)
        end_time = time.time()
        print("Matrix multiplication time (no errors): {:.6f} seconds".format(end_time - start_time))

        # Inject errors into matrix B (you can choose matrix A or C as well)
        error_rate = 0.1
        inject_error(B, error_rate)

        # Matrix multiplication with errors
        start_time = time.time()
        C_with_errors = matrix_multiply(A, B)
        end_time = time.time()
        print("Matrix multiplication time (with errors): {:.6f} seconds".format(end_time - start_time))

        # Check for errors in the whole matrix
        start_time = time.time()
        error_detected = check_errors(A, B, C_with_errors)
        end_time = time.time()

        if error_detected:
            print("Error detected! Detection time: {:.6f} seconds".format(end_time - start_time))
        else:
            print("No errors detected.")

    else:
        A = None
        B = None

    # Broadcast matrices A and B to all processes
    A, B = comm.bcast((A, B), root=0)

    # Distribute matrices among processes
    local_A, local_B = distribute_matrices(A, B, comm)

    # Matrix multiplication within each process
    C_local = matrix_multiply(local_A, local_B)

    # Gather local results to rank 0
    result_matrix = gather_results(C_local, comm)

    # Check for errors in the whole result matrix
    if rank == 0:
        start_time = time.time()
        error_detected = check_errors(A, B, result_matrix)
        end_time = time.time()

        if error_detected:
            print("Error detected in the whole result matrix! Detection time: {:.6f} seconds".format(end_time - start_time))
        else:
            print("No errors detected in the whole result matrix.")

        # Write the result matrix to a CSV file
        result_file_path = "resultMatrixTask2.csv"
        write_matrix_to_csv(result_matrix, result_file_path)
        print("Result matrix written to:", result_file_path)

if __name__ == "__main__":
    main()
