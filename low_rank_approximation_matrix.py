import numpy as np
import math

W = [
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
[1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0] ,
[1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0] ,
[1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1] ,
[0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
]

matrix_graph = np.array(W)

def lanczos_iteration(W):

  #* Membuat baris dan kolom dari W
  row, column = W.shape

  alpha = np.zeros(column+1) #Array Alpha
  beta = np.zeros(column+1) #Array beta
  q_before = 0 #q0

  #* Mencoba salah satu b agar menguak alasan kenapa b = arbitrary
  # b = np.full(column, column)
  b = np.ones(row) #b
  
  q_now = b/np.linalg.norm(b) #q1

  T = np.zeros(row*column).reshape(row, column) #Matriks T
  Q = np.zeros(row*column).reshape(row, column) #Matriks Q
  Q[:, 0] = q_now
  
  print("\n=================\n Q \n=================\n")
  print(np.around(Q, 3))

  #* Proses iterasi lanczos
  for i in range(1, row+1):
    
    q_now = Q[:, i-1][np.newaxis].transpose() #Mengambil suatu kolom di matriks Q, lalu ditranspose agar posisinya tetap jadi kolom.
    v = W.dot(q_now)
    alpha[i] = q_now.transpose().dot(v)

    if(i <= 1):
      v = v - alpha[i]*q_now
    else:
      v = v - beta[i-1]*q_before - alpha[i]*q_now

    #* Memasukkan nilai pada matriks T dan matriks Q
    T[i-1,i-1] = alpha[i] #Memasukkan nilai alpha ke T
    beta[i] = np.linalg.norm(v)
    if(i < row):
      # Dua baris di bawah memasukkan nilai beta ke T
      T[i-1, i] = beta[i]
      T[i, i-1] = beta[i]
      
      # Memasukkan q_now terbaru ke suatu kolom di Q
      q_before = q_now
      q_now = v/beta[i]
      Q[:, i] = q_now.transpose()

  # print("\n=================\n Alpha \n=================\n")
  # print(np.around(alpha, 2))

  # print("\n=================\n Beta \n=================\n")
  # print(np.around(beta, 2))

  # print("\n=================\n T \n=================\n")
  # print(np.around(T, 2))

  # print("\n=================\n Q \n=================\n")
  # print(np.around(Q, 2))

  # print("\n=================\n Q Transpose \n=================\n")
  # print(np.around(Q.transpose(), 2))

  return Q.dot(T).dot(Q.transpose())

W_hat = lanczos_iteration(matrix_graph)
print("\n=================\n W_Hat \n=================\n")
print(np.around(W_hat, 2))