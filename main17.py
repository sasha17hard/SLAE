import numpy as np
from numpy import linalg as LA


def kr_kap(coef):
    # определение основной и расширенной матриц системы

    A = np.delete(coef, 2, 1)  # основная матрица
    B = coef  # расширенная матрица

    # проверяем теорему Кронекера-Капелли (о равенстве двух рангов матриц)

    if LA.matrix_rank(A) == LA.matrix_rank(B):
        print('У системы есть решение. Приступаю к поиску.')
    else:
        print('Система не совместна')


def normir(matrix, i, j):
    if matrix[i][i] == 0:
        return False
    else:
        if matrix[i][i] != 1:
            matrix[i][:] = matrix[i][:] / matrix[i][i]
        matrix_pr = matrix.copy()

        for i in range(len(matrix[0]) - 1):
            if i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return matrix, matrix_pr


def recount(matrix, i):
    for r in range(0, len(matrix)):
        for c in range(0, len(matrix[0])):
            if r != i and c != i:
                matrix[r][c] = matrix[r][c] * matrix[i][i] - matrix[r][i] * matrix[i][c]
    return matrix


def replacement(A, B, i):
    for r in range(0, len(A)):
        for c in range(0, len(A[0])):
            if r != i and c != i and A[r][c] != B[r][c]:
                A[r][c] = B[r][c]
    return A


def output(matrix):
    for i in range(len(matrix)):
        answer = 'x_' + str(i + 1) + ' = ' + str(matrix[i][-1])
        print(answer)
    return ('Решения найдены')


# входные данные(коэффициенты при неизвестных переменных + свободные члены)

# coef = np.array([[2, 3, 12], [3, -1, 7]], dtype=float)
coef = np.array([[3, -2, 5, 7], [7, 4, -8, 3], [5, -3, -4, -12]], dtype=float)

# проверяем теорему Кронекера-Капелли (о равенстве двух рангов матриц)

kr_kap(coef)

# реализация решения

for step in range(len(coef)):
    N = normir(coef, step, step)
    if N:
        R = recount(N[1], step)
        Answer = replacement(N[0], R, step)
        print(Answer)
print(output(Answer))
