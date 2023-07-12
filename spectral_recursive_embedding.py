import numpy as np
import scipy as sp

import normalized_laplacian as nl
import matrix_processing as mp
import the_moment as tm

# tagdoc = np.array([
#   [1, 1, 1, 0, 0],
#   [1, 0, 1, 0, 0],
#   [0, 1, 1, 0, 0],
#   [0, 0, 1, 1, 1]
# ])

# docword = np.array([
#   [0, 1, 1, 0, 0, 0],
#   [1, 1, 0, 1, 1, 1],
#   [1, 0, 1, 1, 0, 0],
#   [1, 0, 0, 0, 1, 1],
#   [0, 0, 0, 0, 1, 1]
# ])
# W = np.array([
#   [0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
#   [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
#   [1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0] ,
#   [1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0] ,
#   [1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1] ,
#   [0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] ,
#   [0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0] ,
#   [0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
#   [0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
# ])
# W = mp.matrixABtoW(tagdoc, docword)

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

  while k <= 10:
    v = p / beta
    k += 1
    r = A.dot(v) - beta*u
    alpha = np.linalg.norm(r)
    u = r / alpha
    p = A.transpose().dot(u) - alpha*v
    beta = np.linalg.norm(p)
    
    u_list.append(u)
    v_list.append(v)
    
  return u_list[-2], v_list[-2]

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
  D_inverse_05 = sp.linalg.fractional_matrix_power(D, -0.5)
  # D_inverse_05 = np.linalg.inv(D_05)
  
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
      XY: Tanda untuk pada vertex keberapa ia di klaster ini
  """
  
  XY = list(set(X).union(set(Y)))
  len_xy = len(XY)
  matrix = np.zeros((len_xy , len_xy))
  
  for i in range(0, len_xy):
    xy_i = XY[i]
    for j in range(0, len_xy):
      xy_j = XY[j]
      matrix[i][j] = W[xy_i][xy_j]
  
  return matrix, XY

def spectral_recursive_embedding(W_hat, W):
  """
    Proses keseluruhan dari Spectral Recursive Embedding

    Args:
      W_hat: Matrix W_hat
      W: Matrix W
      
    Returns:
    G_AB, G_AcBc = Hasil matrix dari SRE
    C1, C2 = Klaster
  """
  tm.this_moment("Start :")
  u, v = singular_vector(W_hat)
  tm.this_moment("Find Second Largest Singular Vector :")
  A_partition, Ac_partition, B_partition, Bc_partition = partition(u, v, W)
  tm.this_moment("Partition The Vertex of Matrix :")
  G_AB, C1 = create_matrix_from_two_vertex(A_partition, B_partition, W)
  G_AcBc, C2 = create_matrix_from_two_vertex(Ac_partition, Bc_partition, W)
  tm.this_moment("Fusion The Vertex :")
  return (G_AB, C1), (G_AcBc, C2) 