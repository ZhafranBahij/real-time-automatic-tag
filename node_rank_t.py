import numpy as np

import matrix_processing as mp

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

fake_matrix_a = np.array([
  [1, 1, 1, 0, 0],
  [1, 0, 1, 0, 0],
  [0, 1, 1, 0, 0],
  [0, 0, 1, 1, 1]
])
fake_matrix_b = np.array([
  [0, 1, 1, 0, 0, 0],
  [1, 1, 0, 1, 0, 0],
  [1, 0, 1, 1, 0, 0],
  [1, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 1, 1]
])
fake_matrix_w = mp.matrixABtoW(fake_matrix_a, fake_matrix_b)

fake_matrix_a1 = np.array([
  [1, 1, 1],
  [1, 0, 1],
  [0, 0, 1],
])
fake_matrix_b1 = np.array([
  [0, 1, 1, 0],
  [1, 1, 0, 1],
  [1, 0, 1, 1]
])
fake_matrix_w1 = mp.matrixABtoW(fake_matrix_a1, fake_matrix_b1)

fake_matrix_a2 = np.array([
  [1, 1]
])
fake_matrix_b2 = np.array([
  [1, 1],
  [1, 1]
])
fake_matrix_w2 = mp.matrixABtoW(fake_matrix_a2, fake_matrix_b2)

all_fake_matrix_partition = [fake_matrix_w1, fake_matrix_w2]
# matrix_graph = np.array(W)

fake_tag_list = [
  ['tag01', [1], [0, 0]],
  ['tag02', [1], [0, 1]],
  ['tag03', [1], [0, 2]],
  ['tag04', [2], [0, 0]],
]

def n_precision(matrix_w_partition, node_i):
  # contoh menggunakan tag1 untuk np
  npi_top = sum(matrix_w_partition[node_i])
  npi_bottom = sum(sum(matrix_w_partition))/2 
  npi = npi_top / npi_bottom
  return npi

def n_recall(matrix_w_origin, tag_cluster):
  # contoh menggunakan tag1 nr
  nri_top = sum(matrix_w_origin[0]) 
  nri_bottom = len(tag_cluster)
  nri = nri_top / nri_bottom
  return nri

def rank(npi, nri):
  ri = npi * np.log10(nri)
  ranki = 0
  if (ri != 0):
    ranki =  np.exp(-1 / ri)
  
  return ranki

def node_rankt(tag_list, matrix_w_original, all_matrix_partition):
  
  all_tag_list_with_rank = []
  
  for tag, clusters, nodes in tag_list:
    nr = n_recall(matrix_w_original, clusters)
    for k in clusters:
      np = n_precision(all_matrix_partition[k], nodes[k])
      ranki = rank(np, nr)
  
  return all_tag_list_with_rank

npi_t1 = n_precision(fake_matrix_w)
nri_t1 = n_recall(fake_matrix_w, fake_tag_list[0])
ranki_t1 = rank(npi_t1, nri_t1)
print("Yahooo")