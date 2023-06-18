import numpy as np
import scipy as sp

# W = [
# [0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
# [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
# [1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0] ,
# [1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0] ,
# [1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1] ,
# [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] ,
# [0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0] ,
# [0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
# [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
# ]

# matrix_graph = np.array(W)

def diagonal_matrix(W):
    """
        Fungsi untuk mmembuat matriks diagonal D 

        Args:
          W: Matriks W

        Returns:
          D: Matriks diagonal D
    """

    row, column = W.shape # Membuat row dan column
    # print(W.shape)
    D = np.zeros(row*column).reshape(row, column)

    # Memasukkan value ke matriks D
    di = W.sum(axis=1)
    for i in range(0, row):
      D[i, i] = di[i]
    
    # print("Matriks D \n")
    # print(D)
    return D

def normalized_laplacian(D, W):
    """
      Melakukan algoritma normalized laplacian

      Args:
        W: Matriks W
        D: Matriks diagonal D

      Returns:
        LW: 
    """
    D = sp.linalg.fractional_matrix_power(D, 0.5)
    D = np.linalg.inv(D)
    LW = D.dot(W).dot(D)

    return LW

# matrix_d = diagonal_matrix(matrix_graph)
# matrix_lw = normalized_laplacian(matrix_d, matrix_graph)
# print(matrix_lw)
# A = np.array([
#   [1, 2],
#   [2, 1]
# ]) 

# B = np.array([
#   [3, 4],
#   [4, 3]
# ]) 