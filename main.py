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
import two_way_poisson_mixture_model as twpmm
import word_count_in_list as wcil

import numpy as np

tm.this_moment("Menjalankan Algoritma :")

# Mengambil data dari database
dataset_document = dfd.get_data()
tm.this_moment("Mengambil dataset :")

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word, title_id_document, all_tag_list, all_word_list, dataframe_document_tag, dataframe_document_word = ip.document_processing(dataset_document)
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
for matrix_partition in all_matrix_partition:
  Q, T = lram.lanczos_iteration(matrix_partition, 1)
  matrix_W_hat_partition = lram.low_rank_approximation_matrix(Q, T)
  all_matrix_w_hat_partition.append(matrix_W_hat_partition)
tm.this_moment("Create W hat partition :")

all_tag_list_with_cluster, all_title_id_document_with_cluster, all_word_list_with_cluster = al.assign_label_cluster(title_id_document, all_tag_list, all_word_list, all_cluster)
tm.this_moment("Assign Label :")

all_tag_list_with_rank = nrt.node_rankt(all_tag_list_with_cluster, matrix_w, all_matrix_partition)
tm.this_moment("Node Rank T :")

# Menghitung banyaknya document dari 
all_title_id_document_with_word_count, total_doc, total_doc_in_cluster = wcim.word_count_in_matrix(all_title_id_document_with_cluster, all_matrix_partition)
tm.this_moment("Word Count setiap Document dari Matrix partisi :")
# print("X")

# Menghitung banyaknya word serta banyaknya word di masing-masing klaster
all_word_list_with_count, total_word, total_word_in_cluster = wcil.word_count_in_list(all_word_list_with_cluster, matrix_w, all_matrix_partition)
tm.this_moment("Word Count setiap word :")

# Two Way Poisson Mixture Model
# Menghitung prior probability
all_prior_probability_m = twpmm.first_prior_probability(total_word, total_word_in_cluster)
tm.this_moment("prior probability :")
# Menghitung nilai lambda
all_word_list_with_lambdamj = twpmm.lambda_m_j_list(all_word_list_with_count, total_doc_in_cluster)
tm.this_moment("lambda(m,j) :")
# Menghitung nilai p(i,m)
all_title_id_document_with_p_im = twpmm.p_im_list(all_title_id_document_with_word_count, all_prior_probability_m, all_word_list_with_lambdamj, dataframe_document_word)
tm.this_moment("p(i,m) :")
# Menghitung nilai likelihood
L = twpmm.get_L(all_title_id_document_with_p_im)
tm.this_moment("Menghitung nilai Log Likelihood :")
print("X")
