# Hanya berisi data palsu untuk mempermudah uji coba
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
import word_count_in_matrix as wcim
import two_way_poisson_mixture_model as twpmm
import word_count_in_list as wcil

# x = np.array([1, 2, 3, 4, 5])
# dotx = np.prod(x)

# factorial_val = np.prod(np.arange(1, 5+1))
# print(x)

K = 2
M = 4
L = 2

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
fake_dataframe_b = pd.DataFrame({
  0: [0, 1, 1, 0, 0, 0],
  1: [1, 1, 0, 1, 0, 0],
  2: [1, 0, 1, 1, 0, 0],
  3: [1, 0, 0, 0, 1, 1],
  4: [0, 0, 0, 0, 1, 1]
}, index=['word1', 'word2', 'word3', 'word4', 'word5', 'word6']).transpose()
fake_matrix_w = mp.matrixABtoW(fake_matrix_a, fake_matrix_b)

# a = fake_dataframe_b.loc['1']

# for aa in a:
#   xxx = aa

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

fake_title_id_list = [
  [['doc1', 0], [1], [4, 3]],
  [['doc2', 1], [1], [5, 4]],
  [['doc3', 2], [1], [6, 5]],
  [['doc4', 3], [2], [7, 1]],
  [['doc5', 4], [2], [8, 2]],
]

fake_word_list = [
  ['word1', [1], [9, 6]],
  ['word2', [1], [10, 7]],
  ['word3', [1], [11, 8]],
  ['word4', [1], [12, 9]],
  ['word5', [2], [13, 3]],
  ['word6', [2], [14, 4]],
]


all_fake_cluster_word_index = [[4, 5, 6], [7, 8]]

# Contoh menghitung rank T menggunakan fake tag list dan fake matrix
all_tag_list_with_rank = nrt.node_rankt(fake_tag_list, fake_matrix_w, all_fake_matrix_partition)

# tag_rank_list = trfnd.tag_recommendation(all_tag_list_with_rank, all_fake_cluster, all_fake_cluster_word_index)
# print("END RANKT")
all_doc_list_with_word_count, fake_total_doc, fake_total_doc_in_cluster = wcim.word_count_in_matrix(fake_title_id_list, all_fake_matrix_partition, fake_dataframe_b)

# fake_total_word dan fake_total_word_in_cluster mungkin bisa berbeda karena ada yg terpotong akibat bipartite graph partition
all_word_list_with_count, fake_total_word, fake_total_word_in_cluster = wcil.word_count_in_list(fake_word_list, fake_matrix_w, all_fake_matrix_partition)

# all_prior_probability_m = twpmm.first_prior_probability(all_doc_list_with_word_count, 2)
all_doc_list_with_m_component, fake_total_doc_in_component = twpmm.set_m_component_to_document(all_doc_list_with_word_count, M ,K)
all_word_list_with_count = twpmm.set_word_count_in_every_m(all_doc_list_with_m_component, all_word_list_with_count, M, K, fake_matrix_b)
all_prior_probability_m = twpmm.first_prior_probability(fake_total_doc, fake_total_doc_in_component)
all_word_list_with_lambdamj = twpmm.lambda_m_j_list(all_word_list_with_count, fake_total_doc_in_component)
fake_doc_list_with_probabililty = twpmm.probability(all_doc_list_with_m_component, all_prior_probability_m, all_word_list_with_lambdamj, fake_dataframe_b, M)
fake_doc_list_with_p_im = twpmm.p_im_list(fake_doc_list_with_probabililty, all_prior_probability_m, all_word_list_with_lambdamj, fake_dataframe_b, M)
L = []
# L.append(twpmm.get_L(fake_doc_list_with_p_im))

for i in range(1, 5):
  all_prior_probability_m, sum_p_im_list = twpmm.pi_m_with_t(fake_doc_list_with_p_im, M)
  all_word_list_with_lambdamj = twpmm.lambda_mt(all_word_list_with_lambdamj, sum_p_im_list,fake_doc_list_with_p_im, M)
  new_fake_doc_list_with_p_im = twpmm.p_im_list_t_more_than_1(fake_doc_list_with_p_im, all_prior_probability_m, all_word_list_with_lambdamj, fake_dataframe_b)
  L.append(twpmm.get_L(fake_doc_list_with_p_im, new_fake_doc_list_with_p_im))
  fake_doc_list_with_p_im = new_fake_doc_list_with_p_im

# fake_10_tag = trfnd.tag_recommendation(all_tag_list_with_rank, all_fake_cluster, all_fake_cluster_word_index, fake_doc_list_with_p_im[1][4][0])
fake_doc_list_with_tag_recommend = trfnd.tag_recommendation_mass(fake_doc_list_with_probabililty, all_tag_list_with_rank, all_fake_cluster, fake_total_doc_in_cluster)
print("END")