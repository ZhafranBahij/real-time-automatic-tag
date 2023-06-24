# Source dari file
import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram

# Source library lain
import datetime

now = datetime.datetime.now()
print("Inisialisasi :",now)

# Mengambil data dari database
dataset_document = dfd.get_data()
now = datetime.datetime.now()
print("Mengambil dataset :",now)
# print(dataset_document)

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word, title_id_document = ip.document_processing(dataset_document)
now = datetime.datetime.now()
print("Dataset ke Matrix :",now)
# print(matrix_tag_document)
# print(matrix_document_word)

matrix_w = mp.matrixABtoW(matrix_tag_document, matrix_document_word)
now = datetime.datetime.now()
print("Matrix A & B menjadi W :",now)
# print(matrix_w)

# matrix_d = nl.diagonal_matrix(matrix_w)
# matrix_lw = nl.normalized_laplacian(matrix_d, matrix_w)
# now = datetime.datetime.now()
# print("Normalized Laplacian :",now)
# print(matrix_lw)

matrix_Q, matrix_T = lram.lanczos_iteration(matrix_w, 1)
matrix_W_hat = lram.low_rank_approximation_matrix(matrix_Q, matrix_T)
now = datetime.datetime.now()
print("Low Rank Approximation :",now)
# print(matrix_W_hat)

matrix_Q_2, matrix_T_2 = lram.lanczos_iteration(matrix_w, 0)
matrix_W_hat_2 = lram.low_rank_approximation_matrix(matrix_Q_2, matrix_T_2)
now = datetime.datetime.now()
print("Low Rank Approximation :",now)
# print(matrix_W_hat_2)

matrix_W_hat_diff = matrix_W_hat - matrix_W_hat_2
print(matrix_W_hat_diff)
