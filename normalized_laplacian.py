import numpy as np

def diagonal_matrix(W):
    """
        Fungsi untuk mmembuat matriks diagonal D 

        Args:
          W: Matriks W

        Returns:
          D: Matriks diagonal D
    """

    row, column = W.shape # Membuat row dan column
    print(W.shape)
    D = np.zeros(row*column).reshape(row, column)

    # Memasukkan value ke matriks D
    di = W.sum(axis=1)
    for i in range(0, row):
      D[i, i] = di[i]
    
    print("Matriks D \n")
    print(D)
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
    D = np.linalg.inv(np.sqrt(D))
    LW = D.dot(W).dot(D)

    return LW
  
# A = np.array([
#   [1, 2],
#   [2, 1]
# ]) 

# B = np.array([
#   [3, 4],
#   [4, 3]
# ]) 