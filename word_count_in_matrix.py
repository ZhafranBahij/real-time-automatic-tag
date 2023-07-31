import numpy as np


def word_count_in_matrix(document_list, all_matrix_partition):
  """
    Melakukan perhitungan berapa banyak word di doc dari matrix hasil partisi
  """
  new_document_list = []
  
  for title_and_id, cluster, nodes in document_list:
    index_cluster = 0
    word_count_list = []
    for k in cluster:
      row_doc = all_matrix_partition[k-1][nodes[index_cluster + 1]]
      word_count = sum(row_doc[int(len(row_doc)/2):])
      word_count_list.append(word_count)
    
    new_document_list.append([title_and_id, cluster, nodes, word_count_list])
    
  return new_document_list
  