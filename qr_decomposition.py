import numpy as np


def qr_decomposition(matrix_a):
    matrix_r = matrix_a.copy()
    matrix_q = np.eye(matrix_a.shape[0])
    for i in range(matrix_a.shape[0] - 1):
        for j in range(i + 1, matrix_a.shape[0]):
            matrix_q_temp = np.eye(matrix_a.shape[0])
            s = -matrix_r[j, i] / np.sqrt(matrix_r[i, i] ** 2 + matrix_r[j, i] ** 2)
            c = matrix_r[i, i] / np.sqrt(matrix_r[i, i] ** 2 + matrix_r[j, i] ** 2)
            matrix_q_temp[i, i] = c
            matrix_q_temp[j, i] = s
            matrix_q_temp[j, j] = c
            matrix_q_temp[i, j] = -s

            matrix_r = np.dot(matrix_q_temp, matrix_r)
            matrix_q_temp[j, i], matrix_q_temp[i, j] = matrix_q_temp[i, j], matrix_q_temp[j, i]
            matrix_q = np.dot(matrix_q, matrix_q_temp)

    print('Q\n', matrix_q, '\n')
    print('R\n', matrix_r, '\n')
    print('Checking answer\n')
    print('A-QR\n', matrix_a - np.dot(matrix_q, matrix_r), '\n')
    return matrix_q, matrix_r


def system_solution(matrix_r, matrix_q, matrix_a, b):
    matrix_q_transp = np.transpose(matrix_q)
    x = np.linalg.solve(matrix_r, np.dot(matrix_q_transp, b))

    print('x\n', x, '\n')
    print('Checking answer\n')
    print('Ax - b =')
    print(np.dot(matrix_a, x) - b)


print('Input dimension: ')

# Create random matrix A
dimension = int(input())
matrix_a = np.random.rand(dimension, dimension)
matrix_a = matrix_a.astype('float64')
np.set_printoptions(suppress=5)
print('Source matrix (A) \n', matrix_a, '\n')

matrix_q, matrix_r = qr_decomposition(matrix_a)
# Generate free terms
b = np.random.rand(dimension, 1)

system_solution(matrix_r, matrix_q, matrix_a, b)
