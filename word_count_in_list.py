import numpy as np

# Versi di mana word yg di cluster dihitung menggunakan jumlah di matrix_origin
def word_count_in_list(word_list, matrix_origin, all_matrix_partition):
  """
    Menghitung banyaknya word dalam 

  Args:
    document_list: Daftar list dokumen
    all_matrix_partition: sebuah list yang berisi mengenai matriks-matriks yang telah dipartisi
    
  Returns:
    new_document_list: dokumen list dengan tambahan banyaknya kata dalam satu dokumen
    total_doc: menyimpan hasil berupa jumlah seluruh dokumen yg ada
    total_doc_per_cluster: banyaknya dokumen per klaster 
  """
  new_word_list = [] 
  total_word = 0 # Menyimpan total word yg ada
  total_word_per_cluster = np.zeros(len(all_matrix_partition)) # Menyimpan total word di masing-masing klaster
  
  # Looping word list
  # name = nama word
  # cluster = word tersebut termasuk klaster berapa
  # indexes = lokasi row index pada suatu matrix
  for name, cluster, indexes in word_list:
    sum_word = []
    
    # Menghitung banyaknya word yang ada di seluruh dokumen
    total_of_this_word = sum(matrix_origin[indexes[0]]) # Menghitung seluruh kata tersebut di dalam matrix
    sum_word.append(total_of_this_word)
    total_word += total_of_this_word
    
    index_cluster = 0
    for k in cluster:
      # total_of_this_word_in_cluster = sum(all_matrix_partition[k-1][indexes[index_cluster+1]])
      # total_of_this_in_cluster = sum(matr)
      sum_word.append(total_of_this_word)
      
      total_word_per_cluster[k-1] += total_of_this_word
      
    new_word_list.append([name, cluster, indexes, sum_word])
    
  return new_word_list, total_word, total_word_per_cluster






