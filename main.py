# Source dari file
import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram
import spectral_recursive_embedding as sre
import the_moment as tm
import assign_label as al
import node_rank_t as nrt
import word_count_in_matrix as wcim

import numpy as np

tm.this_moment("Menjalankan Algoritma :")

# Mengambil data dari database
dataset_document = dfd.get_data()
tm.this_moment("Mengambil dataset :")

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word, title_id_document, all_tag_list, all_word_list = ip.document_processing(dataset_document)
tm.this_moment("dataset ke matrix :")

matrix_w = mp.matrixABtoW(matrix_tag_document, matrix_document_word)
tm.this_moment("matrix A & B menjadi W :")

# matrix_d = nl.diagonal_matrix(matrix_w)
# matrix_lw = nl.normalized_laplacian(matrix_d, matrix_w)
# now = datetime.datetime.now()
# print("Normalized Laplacian :",now)
# print(matrix_lw)

matrix_Q, matrix_T = lram.lanczos_iteration(matrix_w, 1)
matrix_W_hat = lram.low_rank_approximation_matrix(matrix_Q, matrix_T)
tm.this_moment("Low Rank Approximation :") 
# print(matrix_W_hat)

all_matrix_partition, all_cluster = sre.spectral_recursive_embedding(matrix_W_hat, matrix_w)
tm.this_moment("Spectral Recursive Embedding :")
# print("X")

all_matrix_w_hat_partition = []
for matrix_w in all_matrix_partition:
  Q, T = lram.lanczos_iteration(matrix_w, 1)
  matrix_W_hat_partition = lram.low_rank_approximation_matrix(Q, T)
  all_matrix_w_hat_partition.append(matrix_W_hat_partition)
tm.this_moment("Create W hat partition :")

all_tag_list_with_cluster, all_title_id_document_with_cluster, all_word_list_with_cluster = al.assign_label_cluster(title_id_document, all_tag_list, all_word_list, all_cluster)
tm.this_moment("Assign Label :")

all_tag_list_with_rank = nrt.node_rankt(all_tag_list_with_cluster, matrix_w, all_matrix_partition)
tm.this_moment("Node Rank T :")

# Menghitung banyaknya document dari 
all_title_id_document_with_word_count = wcim.word_count_in_matrix(all_title_id_document_with_cluster, all_matrix_partition)
tm.this_moment("Word Count setiap Document dari Matrix partisi :")
print("X")

# matrix_Q_2, matrix_T_2 = lram.lanczos_iteration(matrix_w, 0)
# matrix_W_hat_2 = lram.low_rank_approximation_matrix(matrix_Q_2, matrix_T_2)
# tm.this_moment("Low Rank Approximation :")
# print(matrix_W_hat_2)

# matrix_W_hat_diff = np.absolute(matrix_W_hat - matrix_W_hat_2)
# print(matrix_W_hat_diff)
