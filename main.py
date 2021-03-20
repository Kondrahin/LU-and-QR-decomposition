# !{sys.executable} -m pip install numpy
import math
import numpy as np


def lu_transformation(A: np.ndarray):
    print('LU transformation \n')
    u = A.copy()
    l = np.eye(A.shape[0])
    pl = [x for x in range(u.shape[0])]
    permutations = 0
    # Go throw the rows
    for i in range(u.shape[0] - 1):
        j = i - 1
        while True:
            j += 1
            if j == u.shape[0]:
                print('It\'s degenerate matrix!')
                exit(0)
            if u[j, i] != 0:
                break

        if j != i:
            # Doing a permutation of the rows in the matrix u
            u[[i, j], :] = u[[j, i], :]
            # Doing a permutation of the indexes in the array of permutations
            pl[i], pl[j] = pl[j], pl[i]
            permutations += 1

        # Nullify the elements under the main element
        for j in range(i + 1, u.shape[0]):
            l[j, i] = u[j, i] / u[i, i]
            u[j][i:] = u[j][i:] - (u[i][i:] * l[j, i])

    print('L \n', l, '\n')
    print('U \n', u, '\n')
    print('pl (array of permutations) \n', pl, '\n')

    check_answer_A = A
    check_answer_pl = pl

    print('Checking answer \n')
    for i in range(len(check_answer_pl)):
        if i != check_answer_pl[i]:
            check_answer_A[[i, check_answer_pl[i]], :] = check_answer_A[[check_answer_pl[i], i], :]
            temp = check_answer_pl[i]
            check_answer_pl[i], check_answer_pl[temp] = check_answer_pl[temp], check_answer_pl[i]

    print('LU \n', np.dot(l, u), '\n')
    print('PA \n', check_answer_A, '\n')
    print('||LU-PA|| \n', np.sum(np.square(np.dot(l, u) - check_answer_A)), '\n')

    return l, u, pl, A, permutations


def calculating_determinant(permutations, u):
    determinant = (-1) ** permutations
    for i in range(u.shape[0]):
        determinant *= u[i, i]
    print('Determinant of A \n', determinant, '\n')
    return determinant


def system_solution(l, pl, u, b):
    print('System solution \n')

    # Ly = Pb solution
    for i in range(len(pl)):
        if i != pl[i]:
            b[[i, pl[i]], :] = b[[pl[i], i], :]
            temp = pl[i]
            pl[i], pl[temp] = pl[temp], pl[i]

    y = np.linalg.solve(l, b)
    print('y \n', y, '\n')

    # Ux = y solution
    x = np.linalg.solve(u, y)
    print('x \n', x, '\n')

    print('Checking answer \n')
    print('Ax - b = ', np.dot(A, x) - b, '\n')

    return x, y


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


print('Input dimension: ')

# Create random matrix A
dimension = int(input())
matrix = np.random.rand(dimension, dimension)
print('Source matrix (A) \n', matrix, '\n')

# LU transformation
l, u, pl, A, permutations = lu_transformation(A=matrix)

# Calculating determinant
calculating_determinant(permutations=permutations, u=u)

# Finding solutions to a system of linear equations
b = np.random.rand(dimension, 1)
print('b \n', b, '\n')
x, y = system_solution(l=l, pl=pl, u=u, b=b)

# Calculating inverse matrix
inverse_A = inverse_matrix(A=A)

# Calculating condition number
condition_number = calculating_condition_number(A=A, inverse_A=inverse_A)
