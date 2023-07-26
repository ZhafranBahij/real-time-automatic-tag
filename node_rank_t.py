import numpy as np

import matrix_processing as mp

def n_precision(matrix_w_partition, node_i_partition):
  """
    Menghitung N Precision dari suatu tag

    Args:
      matrix_w_partition: matrix yg sudah di-partition
      node_i_partition: node index dari tag di dalam matrix partition

    Returns:
      npi: N Precision dari suatu tag
  """

  npi_top = sum(matrix_w_partition[node_i_partition])
  npi_bottom = sum(sum(matrix_w_partition))/2
  npi = npi_top / npi_bottom
  return npi

# def n_recall(matrix_w_origin, tag_cluster, node_i_origin):
#   """
#     Menghitung N Precision dari suatu tag

#     Args:
#       matrix_w_origin: matrix w yg dari awal dibuat
#       tag_cluster: 
#       node_i_origin: node di matrix w yg dari awal dibuat

#     Returns:
#       nri: N Recall dari suatu tag
#   """
#   nri_top = sum(matrix_w_origin[node_i_origin]) 
#   nri_bottom = len(tag_cluster)
#   nri = nri_top / nri_bottom
#   return nri

def n_recall(matrix_w_origin, matrix_w_partition, node_i_origin):
  """
    Menghitung N Precision dari suatu tag

    Args:
      matrix_w_origin: matrix w yg dari awal dibuat
      tag_cluster: 
      node_i_origin: node di matrix w yg dari awal dibuat

    Returns:
      nri: N Recall dari suatu tag
  """
  
  row, col = matrix_w_partition.shape
  nri_top = sum(matrix_w_origin[node_i_origin]) 
  nri_bottom = row
  nri = nri_top / nri_bottom
  return nri

def rank(npi, nri):
  """
    Menghitung Rank T

    Args:
      npi: N Precision dari suatu tag
      nri: N Recall dari suatu tag

    Returns:
      ranki: Rank dari suatu tag
  """
  ri = npi * np.log(nri)
  ranki = 0
  if (ri != 0):
    ranki =  np.exp(-1 / np.power(ri, 2))
  
  return ranki


def node_rankt(tag_list, matrix_w_original, all_matrix_partition):
  """
    Menghitung seluruh nilai Rank T yg ada di dalam tag_list

    Args:
      tag_list: Kumpulan tag
      matrix_w_original: Matrix W awal
      all_matrix_partition: list dari matrixs W yg telah dipartisi

    Returns:
      all_tag_list_with_rank: Tag list yg telah ada nilai Rank T
  """
  all_tag_list_with_rank = []
  
  # Looping seluruh data di tag list
  for tag, cluster, nodes in tag_list:
    
    # nr = n_recall(matrix_w_original, cluster, nodes[0])
    
    # Menghitung np & ranki
    index_cluster = 0
    rank_i_list = []
    np_i_list = []
    for k in cluster:
      np = n_precision(all_matrix_partition[k-1], nodes[index_cluster + 1])
      nr = n_recall(matrix_w_original, all_matrix_partition[k-1], nodes[0])
      ranki = rank(np, nr)
      index_cluster += 1
      np_i_list.append(np)
      rank_i_list.append(ranki)
    
    all_tag_list_with_rank.append([tag, cluster, nodes, rank_i_list, nr, np_i_list])
  
  return all_tag_list_with_rank

