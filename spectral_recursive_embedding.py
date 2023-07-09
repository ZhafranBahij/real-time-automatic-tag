import numpy as np
import scipy as sp

import normalized_laplacian as nl

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


A = np.array(W)

def singular_vector(A):
  """
      Membuat dua vektor singular menggunakan Lanczos Bidiagonalization berdasarkan sumber
      G. H. Golub and C. F. Van Loan, Matrix computations, 3nd ed., Johns Hopkins University Press, Baltimore, Maryland, 1996.
      pada chapter 9.3.3
      
      Penjelasan Singular Vector Kiri dan Kanan
      https://www.sciencedirect.com/topics/mathematics/right-singular-vector

      Args:
        A: Matriks

      Returns:
        u: Singular vector kiri terbesar kedua
        v: Singular vector kanan terbesar kedua
  """

  row, column = A.shape

  vector_all_one = np.ones(row)
  v = vector_all_one / np.linalg.norm(vector_all_one)
  p = v
  beta = 1
  k = 0
  u = 0
  
  v_list = []
  u_list = []

  while np.round(beta, 8) != 0:
    v = p / beta
    k += 1
    r = A.dot(v) - beta*u
    alpha = np.linalg.norm(r)
    u = r / alpha
    p = A.transpose().dot(u) - alpha*v
    beta = np.linalg.norm(p)
    
  return u, v

def partition(svl, svr, W): 
  """
    Membuat dua vektor singular menggunakan Lanczos Bidiagonalization berdasarkan sumber
    G. H. Golub and C. F. Van Loan, Matrix computations, 3nd ed., Johns Hopkins University Press, Baltimore, Maryland, 1996.
    pada chapter 9.3.3
    
    Penjelasan Singular Vector Kiri dan Kanan
    https://www.sciencedirect.com/topics/mathematics/right-singular-vector

    Args:
      svl: singular vektor kiri terluas kedua
      svr: singular vektor kanan terluas kedua
      W: Matrix W sebelum proses W_hat


    Returns:
      u: Singular vector kiri terbesar kedua
      v: Singular vector kanan terbesar kedua
  """
  cx = 0
  cy = 0

  D = nl.diagonal_matrix(W)
  D_05 = sp.linalg.fractional_matrix_power(D, 0.5)
  D_inverse_05 = np.linalg.inv(D_05)
  
  # W_hat = D_inverse_05.dot(W).dot(D_inverse_05)

  x = D_inverse_05.dot(svl)
  y = D_inverse_05.dot(svr)

  A_partition = []
  Ac_partition = []
  B_partition = []
  Bc_partition = []

  i = 0
  for value in x:
    if value >= cx:
      A_partition.append(i)
    else:
      Ac_partition.append(i)
    i+=1

  j = 0
  for value in y:
    if value >= cy:
      B_partition.append(j)
    else:
      Bc_partition.append(j)
    j+=1
  
  return A_partition, Ac_partition, B_partition, Bc_partition

def create_matrix_from_two_vertex(X, Y, W):
  """
    Menggabungkan dua partisi ke dalam satu matrix berdasarkan
    'Bipartite graph partitioning and data clustering'

    Args:
      X: Vertex x
      Y: Vertex y
      W: Matrix W awal

    Returns:
      matrix: Matrix dari bipartite graph yg terbaru
  """
  
  
  XY = list(set(X).union(set(Y)))
  len_xy = len(XY)
  matrix = np.zeros((len_xy , len_xy))
  
  for i in range(0, len_xy):
    xy_i = XY[i]
    for j in range(0, len_xy):
      xy_j = XY[j]
      matrix[i][j] = W[xy_i][xy_j]
  
  return matrix

def spectral_recursive_embedding(W):
  u, v = singular_vector(W)
  A_partition, Ac_partition, B_partition, Bc_partition = partition(u, v, W)
  G_AB = create_matrix_from_two_vertex(A_partition, B_partition, W)
  G_AcBc = create_matrix_from_two_vertex(Ac_partition, Bc_partition, W)
  aaa = 0
  
  return G_AB, G_AcBc

spectral_recursive_embedding(A)

"""
Rangkuman: 
G(A, B) akan dibentuk melalui A_partition, B_partition.
Nanti, setiap A_partition akan disambung ke B_partition.
Endingnya akan membentuk matrix simetris. Cara nyambungnya dengan mencocokan nilai a dan b di matrix W
G(Ac, Bc) akan dibentuk melalui Ac_partition, Bc_partition
matrix W yg berasal dari bipartite graph tuh simetris
"""