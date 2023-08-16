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
import tag_recommendation_for_new_document as trfnd
import top_k_accuracy as tka
import data_testing_converter as dtc

import numpy as np

tm.this_moment("Menjalankan Algoritma :")

# Mengambil data dari database
dataset_document = dfd.get_data()
tm.this_moment("Mengambil dataset :")

# Mensetting K, M, dan L
K = 2
M = 2
L = 2

number_partition_dataset_document = int(len(dataset_document) * 80 / 100)
dataset_train = dataset_document[0:number_partition_dataset_document]
tm.this_moment("Mengambil dataset latih :")

dataset_test = dataset_document[number_partition_dataset_document:]
tm.this_moment("Mengambil dataset uji :")

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word, title_id_document, all_tag_list, all_word_list, dataframe_document_tag, dataframe_document_word = ip.document_processing(dataset_train)
tm.this_moment("dataset ke matrix :")

matrix_w = mp.matrixABtoW(matrix_tag_document, matrix_document_word)
tm.this_moment("matrix A & B menjadi W :")

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
all_title_id_document_with_word_count, total_doc, total_doc_in_cluster = wcim.word_count_in_matrix(all_title_id_document_with_cluster, all_matrix_partition, dataframe_document_word)
tm.this_moment("Word Count setiap Document dari Matrix partisi :")
# print("X")

# Menghitung banyaknya word serta banyaknya word di masing-masing klaster
all_word_list_with_count, total_word, total_word_in_cluster = wcil.word_count_in_list(all_word_list_with_cluster, matrix_w, all_matrix_partition)
tm.this_moment("Word Count setiap word :")

# Two Way Poisson Mixture Model
# Memilih m component
all_title_id_document_with_m_component, total_doc_in_component = twpmm.set_m_component_to_document(all_title_id_document_with_word_count, M ,K)
tm.this_moment("Menentukan m component pada suatu klaster :")

# Menghitung banyaknya word serta banyaknya word di masing-masing komponen
all_word_list_with_count = twpmm.set_word_count_in_every_m(all_title_id_document_with_m_component, all_word_list_with_count, M, K, matrix_document_word)
tm.this_moment("Word Count setiap word :")
# Menghitung prior probability
all_prior_probability_m = twpmm.first_prior_probability(total_doc, total_doc_in_component)
tm.this_moment("prior probability :")
# Menghitung nilai lambda
all_word_list_with_lambdamj = twpmm.lambda_m_j_list(all_word_list_with_count, total_doc_in_component)
tm.this_moment("lambda(m,j) :")
# Menghitung probabilitas
all_title_id_document_with_probability = twpmm.probability(all_title_id_document_with_m_component, all_prior_probability_m, all_word_list_with_lambdamj, dataframe_document_word, M)
tm.this_moment("P(D = d|C = k) :")
# Menghitung nilai p(i,m)
all_title_id_document_with_p_im = twpmm.p_im_list(all_title_id_document_with_probability, all_prior_probability_m, all_word_list_with_lambdamj, dataframe_document_word, M)
tm.this_moment("p(i,m) :")

# Looping Expectation Maximization
log_likelihood = []
for i in range(1, 20):
  # Menghitung Prior probability (pi_m) t+1
  all_prior_probability_m, sum_p_im_list = twpmm.pi_m_with_t(all_title_id_document_with_p_im, M)
  tm.this_moment("pi(m) (t+1) :")
  # Menghitung lambda t+1
  all_word_list_with_lambdamj = twpmm.lambda_mt(all_word_list_with_lambdamj, sum_p_im_list, all_title_id_document_with_p_im, M)
  tm.this_moment("lambda(m) (t+1) :")
  # Menghitung nilai likelihood
  new_all_title_id_document_with_p_im = twpmm.p_im_list_t_more_than_1(all_title_id_document_with_p_im, all_prior_probability_m, all_word_list_with_lambdamj, dataframe_document_word)
  tm.this_moment("p(i,m) (t+1) :")
  log_likelihood.append(twpmm.get_log_likelihood(all_title_id_document_with_p_im, new_all_title_id_document_with_p_im))
  all_title_id_document_with_p_im = new_all_title_id_document_with_p_im
  all_title_id_document_with_p_im = twpmm.set_new_probability_t(all_title_id_document_with_p_im)
  tm.this_moment("Menghitung nilai Log Likelihood :")

# Testing
matrix_tag_document_test, matrix_document_word_test, title_id_document_test, all_tag_list_test, all_word_list_test, dataframe_document_tag_test, dataframe_document_word_test = ip.document_processing(dataset_test)
test_title_id_document = dtc.data_testing(title_id_document_test, dataframe_document_word_test)
test_title_id_document_with_probability = twpmm.probability(test_title_id_document, all_prior_probability_m, all_word_list_with_lambdamj, dataframe_document_word_test, M)
test_title_id_document_with_tag_recommendation =  trfnd.tag_recommendation_mass(test_title_id_document_with_probability, all_tag_list_with_rank, all_cluster, total_doc_in_cluster)
tm.this_moment('Testing: ')

# all_title_id_document_with_tag_recommendation = trfnd.tag_recommendation_mass(all_title_id_document_with_probability, all_tag_list_with_rank, all_cluster, total_doc_in_cluster)
# tm.this_moment('Tag Recommendation: ')

top_k_accuracy_value = tka.top_k_accuracy(test_title_id_document_with_tag_recommendation, dataframe_document_tag_test)
tm.this_moment('Top 6 Accuracy: ')
print("X")
