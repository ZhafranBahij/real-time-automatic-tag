import numpy as np

def normalization_vector(vector):
  vector_power = [np.power(x, 2) for x in vector]
  vector_normalize = np.sqrt(np.sum(vector_power))
  return vector_normalize

def lanczos_iteration(A, one_b = 1):
  """
  Melakukan Lanczos iteration dengan membuat matriks Q dan T
  Kemudian, membuat kedua matriks tersebut menjadi W_hat

  Args:
    A: Inputan dari matriks W
    
  Returns:
    Q: Hasil perkalian matriks Q 
  """
  
  k = 50
  
  row, column = A.shape
  
  # Buat matriks T untuk menampung alpha dan beta
  T = np.zeros((k, k))
  
  # buat matriks Q untuk menampung q_now di setiap looping
  Q = np.zeros((row, k))
  
  beta = 0 
  q_before = 0
  b = "" # Bentuk matriks b secara arbitrary (bebas)
  if (one_b != 1):
    b = np.random.default_rng().random((row, 1))
  else:
    b = np.ones((row, 1)) 
    
  # Matriks q_now adalah matriks yg telah normalisasi
  # panjang q_now = 1, cek panjang q_now dengan np.sqrt(np.sum([x*x for x in q_now]))
  q_now = b / normalization_vector(b)
  
  Q[:, 0] = q_now.transpose()
  
  for i in range(1, k):
    v = A.dot(q_now)
    alpha = q_now.transpose().dot(v)
    alpha = alpha[0][0] # Membuat alpha agar menjadi skalar
    v = v - beta*q_before - alpha*q_now
    beta = normalization_vector(v)
    q_before = q_now
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