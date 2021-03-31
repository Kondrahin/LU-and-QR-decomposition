import math
import numpy as np


def lu_transformation(matrix_a, just_gauss=False):
    print('LU transformation \n')
    u = matrix_a.copy()
    l = np.zeros(matrix_a.shape)
    pp = [x for x in range(u.shape[0])]
    pq = [x for x in range(u.shape[0])]
    permutations = 0
    # Go throw the rows
    for i in range(u.shape[0]):
        result = np.where(u[i:, i:] == np.max(u[i:, i:]))
        max_elem_indexes = list(zip(result[0], result[1]))

        max_i = max_elem_indexes[0][0] + i
        max_j = max_elem_indexes[0][1] + i
        print('max indexes', max_elem_indexes)
        print('u before \n', u, '\n')
        if i != max_i:
            u[[i, max_i], :] = u[[max_i, i], :]
            # Doing a permutation of the indexes in the array of rows permutations
            pp[i], pp[max_i] = pp[max_i], pp[i]
            # Doing a permutation of the indexes in the matrix L
            l[[i, max_i], :] = l[[max_i, i], :]
            permutations += 1
        if i != max_j:
            # print(max_j)
            u[:, [i, max_j]] = u[:, [max_j, i]]
            # Doing a permutation of the indexes in the array of columns permutations
            pq[i], pq[max_j] = pq[max_j], pq[i]
            permutations += 1
        for j in range(i + 1, u.shape[0]):
            l[j, i] = u[j, i] / u[i, i]
            u[j][i:] = u[j][i:] - (u[i][i:] * l[j, i])
        l[i, i] = 1
        print('u after \n', u, '\n')

    # If we need just matrix u
    if just_gauss:
        return u

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


def system_solution(l, u, pp, pq, b, matrix_a):
    print('System solution \n')

    # Ly = Pb solution
    temp_b = b
    temp_b = temp_b[pp, :]

    print('pp', pp)
    print('b', b)
    print('temp_b', temp_b)
    y = np.linalg.solve(l, temp_b)
    print('y \n', y, '\n')

    # Uz = y solution
    z = np.linalg.solve(u, y)
    print('z \n', z, '\n')

    # x = Qz
    print('pq', pq)
    print('z', z)
    z = z[pq, :]
    print('z after', z)

    x = z
    print('Checking answer \n')
    print('Ax', np.dot(matrix_a, x))
    print(x)
    print('Ax - b = ', np.dot(matrix_a, x) - b, '\n')

    return x

def system_solution_degenerate(l, u, pp, pq, b, matrix_a, rank_a):
    print('Finding a particular solution \n')
    x = np.linalg.solve(np.dot(l[:rank_a,:rank_a],u[:rank_a,:rank_a]), b[pp,:][:rank_a,:])
    print('x \n', x, '\n')

    print('Checking answer \n')
    print('Ax - b = ', np.dot(matrix_a[pp,:][:,pq][:rank_a, :rank_a], x) - b[pp][:rank_a, :rank_a], '\n')
    return x


def inverse_matrix(A):
    print('Inverse matrix \n')
    inverse_a = np.linalg.solve(A, np.eye(A.shape[0]))
    print('A^(-1) \n', inverse_a, '\n')

    print('Checking answer \n')
    print('A*A^(-1)', np.dot(A, inverse_a), '\n')
    print('A^(-1)*A', np.dot(inverse_a, A), '\n')

    return inverse_a


def calculating_condition_number(A, inverse_A):
    condition_number_A = math.sqrt(np.sum(np.square(A)))
    condition_number_inverse_A = math.sqrt(np.sum(np.square(inverse_A)))

    condition_number = condition_number_A * condition_number_inverse_A
    print('condition_number', condition_number)
    return condition_number


def calculating_rank(u):
    rank = u.shape[0]
    degenerate = True
    print('u.shape', u.shape[0])
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
    print('Rank degenerate matrix: ', rank)
    return rank


def check_compatible(matrix_a, b, rank_a):
    extended_matrix = np.column_stack((matrix_a, b))
    u_extended = lu_transformation(extended_matrix, just_gauss=True)
    rank_extended = calculating_rank(u_extended)
    if rank_a == rank_extended:
        system_solution_degenerate(l,u,pp,pq,b,matrix_a,rank_a)
    else:
        print('System not compatible')


print('Input dimension: ')

# Create random matrix A
dimension = int(input())
matrix_a = np.random.rand(dimension, dimension)
matrix_a = matrix_a.astype('float64')
np.set_printoptions(suppress=5)
print('Source matrix (A) \n', matrix_a, '\n')

# LU transformation
l, u, pp, pq, permutations = lu_transformation(matrix_a)

# Calculating determinant
determinant = calculating_determinant(permutations, u)

# Generate free terms
b = np.random.rand(dimension, 1)
print('b \n', b, '\n')

if determinant:
    # Finding solutions to a system of linear equations
    x = system_solution(l, u, pp, pq, b, matrix_a)

    # Calculating inverse matrix
    inverse_a = inverse_matrix(matrix_a)

# Calculating condition number
condition_number = calculating_condition_number(matrix_a, matrix_a)
if not determinant:
    # Calculating rank of a degenerate matrix
    rank_a = calculating_rank(u)
    # Check if the system is compatible
    check_compatible(matrix_a, b, rank_a)
