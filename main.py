# Source dari file
import data_from_database as dfd 
import matrix_processing as mp
import input_processing as ip
import normalized_laplacian as nl
import low_rank_approximation_matrix as lram

# Mengambil data dari database
dataset_document = dfd.get_data()
# print(dataset_document)

# Memproses dataset menjadi matrix
matrix_tag_document, matrix_document_word = ip.document_processing(dataset_document)
print(matrix_tag_document)
print(matrix_document_word)

matrix_w = mp.matrixABtoW(matrix_tag_document, matrix_document_word)
# matrix_w = mp.getW()
print(matrix_w)

matrix_lw = nl.normalized_laplacian(nl.diagonal_matrix(matrix_w), matrix_w)
print(matrix_lw)

matrix_Q, matrix_T = lram.lanczos_iteration(matrix_w)
matrix_W_hat = lram.low_rank_approximation_matrix(matrix_Q, matrix_T)
print(matrix_W_hat)
