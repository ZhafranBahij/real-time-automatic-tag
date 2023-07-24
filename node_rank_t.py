import numpy as np

import matrix_processing as mp

def n_precision(matrix_w_partition, node_i_partition):
  # contoh menggunakan tag1 untuk np
  npi_top = sum(matrix_w_partition[node_i_partition])
  npi_bottom = sum(sum(matrix_w_partition))/2
  npi = npi_top / npi_bottom
  return npi

def n_recall(matrix_w_origin, tag_cluster, node_i_origin):
  # contoh menggunakan tag1 nr
  nri_top = sum(matrix_w_origin[node_i_origin]) 
  nri_bottom = len(tag_cluster)
  nri = nri_top / nri_bottom
  return nri

def rank(npi, nri):
  ri = npi * np.log(nri)
  ranki = 0
  if (ri != 0):
    ranki =  np.exp(-1 / np.power(ri, 2))
  
  return ranki

def node_rankt(tag_list, matrix_w_original, all_matrix_partition):
  
  all_tag_list_with_rank = []
  
  # Looping seluruh data di tag list
  for tag, cluster, nodes in tag_list:
    
    # Menghitung nr
    nr = n_recall(matrix_w_original, cluster, nodes[0])
    
    # Menghitung np & ranki
    index_cluster = 0
    rank_i_list = []
    np_i_list = []
    for k in cluster:
      np = n_precision(all_matrix_partition[k-1], nodes[index_cluster + 1])
      ranki = rank(np, nr)
      index_cluster += 1
      np_i_list.append(np)
      rank_i_list.append(ranki)
    
    all_tag_list_with_rank.append([tag, cluster, nodes, rank_i_list, nr, np_i_list])
  
  return all_tag_list_with_rank

