# Hanya beriis data palsu untuk mempermudah uji coba
import numpy as np

import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram
import spectral_recursive_embedding as sre
import the_moment as tm
import assign_label as al
import node_rank_t as nrt

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

# [nama_tag, klaster, [index_row_matrix_awal, index_row_matrix_klaster]]
fake_tag_list = [
  ['tag01', [1], [0, 0]],
  ['tag02', [1], [1, 1]],
  ['tag03', [1], [2, 2]],
  ['tag04', [2], [3, 0]],
]

# Contoh menghitung rank T pada tag01
all_tag_list_with_rank = nrt.node_rankt(fake_tag_list, fake_matrix_w, all_fake_matrix_partition)
print("END")