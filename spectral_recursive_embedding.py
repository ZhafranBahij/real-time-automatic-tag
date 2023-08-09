import numpy as np
import scipy as sp
import normalized_laplacian as nl
import matrix_processing as mp
import low_rank_approximation_matrix as lram

import the_moment as tm

def second_largest_singular_vector(W_hat):
  """
    Menghitung singular vector menggunakan W_hat dari library
    https://docs.scipy.org/doc/scipy/reference/sparse.linalg.svds-lobpcg.html    
    
    Args:
      W_hat: Matrix W_hat

    Returns:
      second_largest_left: Mengambil vektor U
      second_largest_right: Mengambil vektor Vh
  """
  
  U, s, Vh = sp.sparse.linalg.svds(W_hat, solver='lobpcg')
  # U, s, Vh = sp.linalg.svd(W_hat)
  second_largest_left = U[:, 1] # Mengambil left singular vector kedua
  second_largest_right = Vh[1, :] # Mengambil right singular vector kedua
  
  return second_largest_left, second_largest_right

def find_cut_point(singular_vector_left, singular_vector_right ):
  """
    Mencari cut point

    Returns:
      cx: cx
      cy: cy
  """
  
  cx = np.median(singular_vector_left)  
  cy = np.median(singular_vector_right) 
  return cx, cy

def form_partition(cx, cy, x, y):
  """
    Melakukan form partition
    
    Args:
      cx: cut point untuk x
      cy: cut point untuk y
      x: Hasil dari second_largest_left
      y: Hasil dari second_largest_right
      
    Returns:
      A_partition
      Ac_partition
      B_partition
      Bc_partition
  """
  
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
  
  # Looping untuk membuat matrix_w hasil partisi
  for i in range(0, len_xy):
    xy_i = XY[i]
    for j in range(0, len_xy):
      xy_j = XY[j]
      matrix[i][j] = W[xy_i][xy_j]
  
  return matrix, XY

def spectral_recursive_embedding(W_hat, W):
  """
    Melakukan bipartite graph partition dengan menggunakna
    spectral recursive embedidng

    Args:
      W_hat: Matrix W_hat
      Y: Vertex y
      W: Matrix W awal

    Returns:
      all_matrix: List Matrix w hasil klasterisasi
      all_cluster: Klaster
  """
  all_matrix = []
  all_cluster = []
  # tm.this_moment("Start :")
  x, y = second_largest_singular_vector(W_hat)
  # tm.this_moment("Second Largest Singular Vector :")
  cx, cy = find_cut_point(x, y)
  # tm.this_moment("Cut Point :")
  A, Ac, B, Bc = form_partition(cx, cy, x, y)
  # tm.this_moment("Form Partition :")
  matrix, cluster = create_matrix_from_two_vertex(A, B, W)
  all_matrix.append(matrix)
  all_cluster.append(cluster)
  matrix, cluster = create_matrix_from_two_vertex(Ac, Bc, W)
  all_matrix.append(matrix)
  all_cluster.append(cluster)
  # tm.this_moment("Fusion the matrix :")
  
  return all_matrix, all_cluster