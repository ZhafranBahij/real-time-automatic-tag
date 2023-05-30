import numpy as np

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

def lanczos_iteration(W):
  """
  Melakukan Lanczos iteration dengan membuat matriks Q dan T
  Kemudian, membuat kedua matriks tersebut menjadi W_hat

  Args:
    W: Matriks W
    
  Returns:
    Q: Hasil perkalian matriks Q 
  """
  
  A = np.array(W)
  row, column = W.shape
  
  # Buat matriks T untuk menampung alpha dan beta
  T = np.zeros((row, column))
  
  # buat matriks Q untuk menampung q_now di setiap looping
  Q = np.zeros((row, column))
  
  beta = 0 
  q_before = 0
  b = np.ones((row, 1)) # Bentuk matriks b secara arbitrary (bebas)
  
  q_now = b / np.linalg.norm(b)
  
  Q[:, 0] = q_now.transpose()
  
  for i in range(1, row+1):
    v = A.dot(q_now)
    alpha = q_now.transpose().dot(v)
    v = v - beta*q_before - alpha*q_now
    beta = np.linalg.norm(v)
    q_now = v / beta
    
    # Tampung alpha dan beta ke matriks T
    T[i-1, i-1] = alpha
    if i < row:
      T[i-1, i] = beta
      T[i, i-1] = beta
      
      # Tampung nilai q_now ke matriks Q
      Q[:, i] = q_now.transpose()

  return Q, T

def low_rank_approximation_matrix(Q, T):
  """
  Melakukan perkalian matriks Q, T, dan Q transpose

  Args:
    Q: Matriks Q yang didapat dari Lanczos iteration
    T: Matriks T yang didapat dari Lanczos iteration
    
  Returns:
    W_hat: Hasil perkalian matriks Q, T, dan Q transpose
  """
  
  return Q.dot(T).dot(Q.transpose())

# Q, T = lanczos_iteration(matrix_graph)
# W_hat = low_rank_approximation_matrix(Q, T)
# print(np.around(W_hat, 2))