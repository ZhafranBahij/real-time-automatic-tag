# Source dari file
import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram

# Source library lain
import the_moment as tm
import numpy as np

tm.this_moment("Menjalankan Algoritma :")

# Mengambil data dari database
dataset_document = dfd.get_data()
tm.this_moment("Mengambil dataset :")
# print(dataset_document)

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word, title_id_document = ip.document_processing(dataset_document)
tm.this_moment("dataset ke matrix :")
# print(matrix_tag_document)
# print(matrix_document_word)

matrix_w = mp.matrixABtoW(matrix_tag_document, matrix_document_word)
tm.this_moment("matrix A & B menjadi W :")
# print(matrix_w)

# matrix_d = nl.diagonal_matrix(matrix_w)
# matrix_lw = nl.normalized_laplacian(matrix_d, matrix_w)
# now = datetime.datetime.now()
# print("Normalized Laplacian :",now)
# print(matrix_lw)

matrix_Q, matrix_T = lram.lanczos_iteration(matrix_w, 1)
matrix_W_hat = lram.low_rank_approximation_matrix(matrix_Q, matrix_T)
tm.this_moment("Low Rank Approximation :")
# print(matrix_W_hat)

matrix_Q_2, matrix_T_2 = lram.lanczos_iteration(matrix_w, 0)
matrix_W_hat_2 = lram.low_rank_approximation_matrix(matrix_Q_2, matrix_T_2)
tm.this_moment("Low Rank Approximation :")
# print(matrix_W_hat_2)

matrix_W_hat_diff = np.absolute(matrix_W_hat - matrix_W_hat_2)
print(matrix_W_hat_diff)
