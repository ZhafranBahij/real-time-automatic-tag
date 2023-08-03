import numpy as np

# Versi di mana word yg di cluster dihitung menggunakan jumlah di matrix_origin
def word_count_in_list(word_list, matrix_origin, all_matrix_partition):
  """
    Melakukan perhitungan berapa banyak word di doc dari matrix hasil partisi
  """
  new_word_list = [] 
  total_word = 0 # Menyimpan total word yg ada
  total_word_per_cluster = np.zeros(len(all_matrix_partition)) # Menyimpan total word di masing-masing klaster
  
  for name, cluster, indexes in word_list:
    sum_word = []
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



# def word_count_in_list(word_list, matrix_origin, all_matrix_partition):
#   """
#     Melakukan perhitungan berapa banyak word di doc dari matrix hasil partisi
#   """
#   new_word_list = [] 
#   total_word = 0 # Menyimpan total word yg ada
#   total_word_per_cluster = np.zeros(len(all_matrix_partition)) # Menyimpan total word di masing-masing klaster
  
#   for name, cluster, indexes in word_list:
#     sum_word = []
#     total_of_this_word = sum(matrix_origin[indexes[0]]) # Menghitung seluruh kata tersebut di dalam matrix
#     sum_word.append(total_of_this_word)
    
#     total_word += total_of_this_word
    
#     index_cluster = 0
#     for k in cluster:
#       total_of_this_word_in_cluster = sum(all_matrix_partition[k-1][indexes[index_cluster+1]])
#       sum_word.append(total_of_this_word_in_cluster)
      
#       total_word_per_cluster[k-1] += total_of_this_word_in_cluster
      
#     new_word_list.append([name, cluster, indexes, sum_word])
    
#   return new_word_list, total_word, total_word_per_cluster




