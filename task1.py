'''
NAME: VISHAL PALLI
ID: 49042025
COURSE => ECE8379: FAULT TOLERANT COMPUTING

'''
import numpy as np
import pandas as pd
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def inject_error(matrix, error_rate):
    
    errors = np.random.rand(*matrix.shape) < error_rate
    matrix[errors] = np.random.rand(*matrix.shape)[errors]

def check_errors(A, B, C):
    
    AB = np.dot(A, B)
    error_matrix = AB - C
    error_detected = not np.allclose(error_matrix, np.zeros_like(error_matrix))

    return error_detected

def read_matrix_from_csv(file_path):
   
    matrix_df = pd.read_csv(file_path, header=None)
    return matrix_df.values

def write_matrix_to_csv(matrix, file_path):
    
    pd.DataFrame(matrix).to_csv(file_path, index=False, header=False)

def main():
      
    file_path_A = "matrix1.csv"
    file_path_B = "matrix2.csv"

    A = read_matrix_from_csv(file_path_A)
    B = read_matrix_from_csv(file_path_B)

    
    start_time = time.time()
    C = matrix_multiply(A, B)
    end_time = time.time()
    print("Matrix multiplication time (no errors): {:.6f} seconds".format(end_time - start_time))

    
    error_rate = 0.1
    inject_error(B, error_rate)

 
    start_time = time.time()
    C_with_errors = matrix_multiply(A, B)
    end_time = time.time()
    print("Matrix multiplication time (with errors): {:.6f} seconds".format(end_time - start_time))

   
    start_time = time.time()
    error_detected = check_errors(A, B, C_with_errors)
    end_time = time.time()

    if error_detected:
        print("Error detected! Detection time: {:.6f} seconds".format(end_time - start_time))
    else:
        print("No errors detected.")

    # Write the result matrix to a CSV file
    result_file_path = "resultMatrixTask1.csv"
    write_matrix_to_csv(C_with_errors, result_file_path)
    print("Result matrix written to:", result_file_path)

if __name__ == "__main__":
    main()
