# !{sys.executable} -m pip install numpy
import math
import numpy as np


def lu_transformation(matrix_a, just_gauss=False):
    u = matrix_a.copy()
    l = np.zeros(matrix_a.shape)
    pp = [x for x in range(u.shape[1])]
    pq = [x for x in range(u.shape[1])]
    permutations = 0
    # Go throw the rows
    for i in range(u.shape[0]):
        result = np.where(u[i:, i:] == np.max(u[i:, i:]))
        max_elem_indexes = list(zip(result[0], result[1]))

        max_i = max_elem_indexes[0][0] + i
        max_j = max_elem_indexes[0][1] + i
        # print('max indexes', max_elem_indexes)
        # print('u before \n', u, '\n')
        if i != max_i:
            u[[i, max_i], :] = u[[max_i, i], :]
            # Doing a permutation of the indexes in the array of rows permutations
            pp[i], pp[max_i] = pp[max_i], pp[i]
            # Doing a permutation of the indexes in the matrix L
            l[[i, max_i], :] = l[[max_i, i], :]
            permutations += 1
        if i != max_j:
            u[:, [i, max_j]] = u[:, [max_j, i]]
            # Doing a permutation of the indexes in the array of columns permutations
            pq[i], pq[max_j] = pq[max_j], pq[i]
            permutations += 1
        for j in range(i + 1, u.shape[0]):
            l[j, i] = u[j, i] / u[i, i]
            u[j][i:] = u[j][i:] - (u[i][i:] * l[j, i])
        l[i, i] = 1
        # print('u after \n', u, '\n')

    # If we need just matrix u
    if just_gauss:
        return u

    print('LU transformation \n')
    print('L \n', l, '\n')
    print('U \n', u, '\n')

    print('pp (array of permutations of rows) \n', pp, '\n')
    print('pq (array of permutations of columns) \n', pq, '\n')

    print('Checking answer \n')

    check_answer_A = matrix_a
    check_answer_A = check_answer_A[:, pq]
    check_answer_A = check_answer_A[pp, :]
    print('LU \n', np.dot(l, u), '\n')
    print('PAQ \n', check_answer_A, '\n')
    print('||LU-PAQ|| \n', np.sum(np.square(np.dot(l, u) - check_answer_A)), '\n')
    return l, u, pp, pq, permutations


def calculating_determinant(permutations, u):
    determinant = (-1) ** permutations
    for i in range(u.shape[0]):
        determinant *= u[i, i]
    print('Determinant of A \n', determinant, '\n')
    return determinant


def system_solution(l, u, pp, pq, b, matrix_a, rank_a):
    print('Finding a particular solution \n')
    x = np.linalg.solve(np.dot(l[:rank_a, :rank_a], u[:rank_a, :rank_a]), b[pp, :][:rank_a, :])
    print('x \n', x, '\n')

    print('Checking answer \n')
    print('Ax - b = ', np.dot(matrix_a[pp, :][:, pq][:rank_a, :rank_a], x) - b[pp][:rank_a, :rank_a], '\n')
    return x


def calculating_rank(u):
    rank = u.shape[0]
    degenerate = True
    for i in range(u.shape[0] - 1, -1, -1):
        # Not u.shape[1] because we add column, and u.shape[0] has not changed
        for j in range(u.shape[1]):
            if u[i, j] != 0:
                degenerate = False
                break
        if not degenerate:
            break
        else:
            rank -= 1
    return rank


def check_compatible(matrix_a, b, rank_a, l, u, pp, pq):
    extended_matrix = np.column_stack((matrix_a, b))
    u_extended = lu_transformation(extended_matrix, just_gauss=True)
    rank_extended = calculating_rank(u_extended)
    if rank_a == rank_extended:
        system_solution(l, u, pp, pq, b, matrix_a, rank_a)
    else:
        print('System not compatible')


print('Input dimension: ')

# Create random matrix A
# dimension = int(input())
# matrix_a = np.random.rand(dimension, dimension)
# matrix_a = np.array([[1, 2, 3, 4], [2, 5, 3, 5], [6, 7, 5, 9], [2, 4, 6, 8]])
# matrix_a = np.array([[1, 2, 3, 4, 993], [0, -1, 45, 21, 1], [6, 1, 5, 2, 52], [2, 4, 6, 10, 65], [1,2,3,74,3]])
matrix_a = np.array([[1, 4, 0, 0], [0, 3, 1, 1], [0, 1, 4, 1], [2, 8, 0, 0]])
matrix_a = matrix_a.astype('float64')
np.set_printoptions(suppress=5)
print('Source matrix (A) \n', matrix_a, '\n')

# LU transformation
l, u, pp, pq, permutations = lu_transformation(matrix_a)

# Calculating determinant
determinant = calculating_determinant(permutations, u)

# Generate free terms
# Example with compatible system
b = np.array([[1], [4], [0], [2]])

# Example with incompatible system
# b = np.array([[1], [4], [0], [23]])
# b = np.array([[1], [4], [0], [2], [5]])

if determinant:
    # Finding solutions to a system of linear equations
    print('b \n', b, '\n')
    x = system_solution(l, u, pp, pq, b, matrix_a, u.shape[0])

if not determinant:
    # Calculating rank of a degenerate matrix
    rank_a = calculating_rank(u)
    print('Rank degenerate matrix: ', rank_a)
    # Check if the system is compatible
    check_compatible(matrix_a, b, rank_a, l, u, pp, pq)
