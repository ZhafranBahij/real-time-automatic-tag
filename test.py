import numpy as np
import math

W_mini = [
  [0 , 0 , 1 , 1 , 1 ],
  [0 , 0 , 0 , 0 , 1 ],
  [1 , 0 , 0 , 0 , 0 ],
  [1 , 0 , 0 , 0 , 0 ],
  [1 , 1 , 0 , 0 , 0 ],
]

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

"""
Function Normalization Matrix
=> outputnya -> L(W) = Matriks hasil normalisasi Laplacian
=> input = matrix
"""
def normalization(graph):
  row, column = graph.shape
  LW = []
  di = graph.sum(axis=1) 
  dj = graph.sum(axis=0)

  # print(di)
  # print(dj)

  for i in range(0, row):
    for j in range(0, column):
      
      if i == j : #* Kondisi 1
        LW.append(1 - (graph[i][j]/di[i]))
      
      elif graph[i][j] > 0 : #* Kondisi 2
        LW.append(-(graph[i][j]/math.sqrt(di[i]*dj[j])))
      
      else : #* Kondisi 3
        LW.append(0)

  return np.array(LW).reshape(row, column)

"""
Buat matriks D (matriks diagonal)
"""
def diagonal_matrix(graph):
    row, column = graph.shape
    D = np.zeros(row*column).reshape(row, column)

    for i in range(0, row):
      di = graph.sum(axis=1) 
      D[i, i] = di[i]
    
    return D

"""
Buat normalisasi laplacian
"""
def normalized_laplacian(D, W):
    D = np.linalg.inv(np.sqrt(D))
    LW = D.dot(W).dot(D)

    # # Di rounding karena ini hanya buat contoh
    # LW = np.around(LW, decimals=2)    
    return LW

"""
Iterasi Lanczos
Di sini yang di return adalah W_hat
"""
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

"""
Menari singluar vector terluar kedua di kiri dan kanan
"""
def second_largest_left_and_right_singular_vector(W_hat):
  a = np.array([2, 3, 4, 5]).reshape(2,2)

  #* Mencari SVD
  u, s, vh = np.linalg.svd(W_hat, full_matrices=True)

  print("\n=================\n Nilai Singular Vector \n=================\n")
  print(np.around(u, 2))
  print(np.around(vh, 2))

  
  #* Mencari vektor kedua terbesar dari left singular vector
  second_largest_left = u[:, 1]

  #* Mencari vektor kedua terbesar dari right singular vector
  second_largest_right = vh[1, :]

  return second_largest_left, second_largest_right

"""
Mencari titik potong
Melakukan pemisahan 
"""

def cut_points(D, x_hat, y_hat):
  x = D @ x_hat
  y = D @ y_hat

  #* Menggunakan strategi simple untuk menentukan cutpoint
  cx = 0
  cy = 0

  #* Buat partisinya

  A = []
  i = 0
  for x_i in x :
    if x_i >= cx :
      A.append(i)
    i += 1
  A = np.array(A)

  A_c = []
  i = 0
  for x_i in x :
    if x_i < cx :
      A_c.append(i)
    i += 1
  A_c = np.array(A_c)

  B = []
  j = 0
  for y_i in y :
    if y_i >= cy :
      B.append(j)
    j+=1
  B = np.array(B)

  B_c = []
  j = 0
  for y_i in y :
    if y_i < cy :
      B_c.append(j)
    j+=1
  B_c = np.array(B_c)

  return A, A_c, B, B_c

"""
Menari singular vector terluar kedua di kiri dan kanan
melalui Lanczos Algorithm
"""
def second_largest_left_and_right_singular_vector_lanczos(W_hat):

  A = W_hat

  #* Membuat baris dan kolom dari W
  row, column = W_hat.shape
  
  V = np.zeros(row*column).reshape(row, column)
  U = np.zeros(row*column).reshape(row, column)

  b = np.ones(row) #b
  v_now = b/np.linalg.norm(b) #v1
  beta = 1 #beta0 = 1
  k = 0
  p = v_now # p0 = v1
  u = 0 # u0 = 0

  i = 0
  while beta != 0:
    
    v_now = p/beta
    k += 1
    r = A@v_now - beta*u
    alpha = np.linalg.norm(r)
    u = r/alpha
    p = A.T@u - alpha*v_now
    beta = np.linalg.norm(p)

    V[:, i] = v_now.T
    U[:, i] = u.T
    print(i)

    if(i == row-1):
      break

    i += 1

  print(np.around(U,2))
  print(np.around(V,2))

    #* Mencari vektor kedua terbesar dari left singular vector
  second_largest_left = U[:, 1]

  #* Mencari vektor kedua terbesar dari right singular vector
  second_largest_right = V[1, :]

  return second_largest_left, second_largest_right


print("\n=================\n W \n=================\n")
print(matrix_graph)

# print("\n=================\n W \n=================\n")
# print(matrix_graph[:, 1])
# print(matrix_graph.sum(axis=0))
# print(matrix_graph.sum(axis=1))
D = diagonal_matrix(matrix_graph)
# print(D)
# LW = normalized_laplacian(D, matrix_graph)
# print("\n=================\n LW \n=================\n")
# print(np.around(LW, 2))
# print(normalization(matrix_graph))

# b = np.array([1,1,1, 1,1,1, 1,1,1])
# q1 = b/np.linalg.norm(b)
# print(q1)

# z = np.array([1, 1, 1, 1]).reshape(2,2)
# z = z*33

W_hat = lanczos_iteration(matrix_graph)
print("\n=================\n W_Hat \n=================\n")
print(W_hat)

# print("\n=================\n Tester Transpose array \n=================\n")
# a = np.array([5,4])[np.newaxis]
# matoriksu = np.array([2,2,2,2]).reshape(2,2)
# print(a)
# print(a.T)
# print(matoriksu)
# print(matoriksu*a)
# print(matoriksu.dot(a.T))
# print(matoriksu*a.T)

print("\n=================\n Second Largest Left & Right Singular Vector \n=================\n")
# x_hat, y_hat = second_largest_left_and_right_singular_vector(LW)
# print("\n=================\n Second Largest Singular Vector \n=================\n")
# print(np.around(x_hat, 2))
# print(np.around(y_hat, 2))


print("\n=================\n Second Largest Left & Right   Singular Vector \n=================\n")
x_hat, y_hat = second_largest_left_and_right_singular_vector_lanczos(W_hat)

print("\n=================\n A, Ac, B, Bc \n=================\n")
A, A_c, B, B_c = cut_points(D, x_hat, y_hat)
print(np.around(A, 2))
print(np.around(A_c, 2))
print(np.around(B, 2))
print(np.around(B_c, 2))


