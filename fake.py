# Hanya beriis data palsu untuk mempermudah uji coba
import numpy as np
import pandas as pd

import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram
import spectral_recursive_embedding as sre
import the_moment as tm
import assign_label as al
import node_rank_t as nrt
import tag_recommendation_for_new_document as trfnd

# Membuat fake matrix w
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

# Membuat fake matrix w partition
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
all_fake_cluster = [[0, 1, 2, 4, 5, 6, 9, 10, 11, 12], [3, 7, 8, 13, 14]]

"""
Melakukan rank T dengan fake tag list
"""
# [nama_tag, klaster, [index_row_matrix_awal, index_row_matrix_klaster]]
fake_tag_list = [
  ['tag04', [2], [3, 0]],
  ['tag01', [1], [0, 0]],
  ['tag02', [1], [1, 1]],
  ['tag03', [1], [2, 2]],
]

# Contoh menghitung rank T menggunakan fake tag list dan fake matrix
all_tag_list_with_rank = nrt.node_rankt(fake_tag_list, fake_matrix_w, all_fake_matrix_partition)

all_fake_cluster_word_index = [[4, 5, 6], [7, 8]]
tag_rank_list = trfnd.tag_recommendation(all_tag_list_with_rank, all_fake_cluster, all_fake_cluster_word_index)
# print("END RANKT")




print("END")