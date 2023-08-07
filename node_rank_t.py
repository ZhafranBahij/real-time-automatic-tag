import numpy as np

import matrix_processing as mp

# def n_precision(matrix_w_partition, node_i_partition):
#   """
#     Menghitung N Precision dari suatu tag

#     Args:
#       matrix_w_partition: matrix yg sudah di-partition
#       node_i_partition: node index dari tag di dalam matrix partition

#     Returns:
#       npi: N Precision dari suatu tag
#   """

#   npi_top = sum(matrix_w_partition[node_i_partition])
#   npi_bottom = sum(sum(matrix_w_partition))/2
#   npi = npi_top / npi_bottom
#   return npi

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

# Untuk mendapatkan np top di setiap tag
def get_all_np_top_list(tag_list, all_matrix_partition):
  np_top_list = []
  k_list = []
  
  for tag, cluster, nodes in tag_list:
    # nodes = [index_matrix_origin, index_matrix_partition_cluster_1, ...., index_matrix_partition_cluster_n]
    index_cluster = 0
    for k in cluster:
      if not k in k_list:
        start_range = 0
        if len(k_list) > 0:
          start_range = k_list[-1]
        for ik in range(start_range+1, k+1):
          k_list.append(ik)
          np_top_list.append([])
      
      npi_top = sum(all_matrix_partition[k-1][nodes[index_cluster + 1]])
      np_top_list[k-1].append(npi_top)
      index_cluster += 1
  
  return np_top_list

# mendapatkan nilai N Precission di setiap tag
def get_all_np_list(tag_list, np_top_list):
  np_list = []
  k_list = []
  sum_np_top = []
  
  for tag, cluster, nodes in tag_list:
    index_cluster = 0
      
    # Jika klaster k belum ada di k_list
    for k in cluster:
      if not k in k_list:
        start_range = 0
        if len(k_list) > 0:
          start_range = k_list[-1]
        for ik in range(start_range+ 1, k+1):
          k_list.append(ik)
          np_list.append([])
          sum_np_top.append(sum(np_top_list[ik-1]))
    
      # Mengammbil np_top
      np_top_i = np_top_list[k-1][nodes[index_cluster+1]]
      # Menghitung np_bottom
      np_bottom = sum_np_top[k-1]
      
      # alternate menghitung np
      # np_buttom = sum_np_top[k-1] - np_top_i
      # npi = 1
      # if np_bottom != 0:
      #   npi = np_top_i / np_bottom

      
      # Menghitung np pada suatu tag
      npi = np_top_i / np_bottom
      
      # Menampung nilai np
      np_list[k-1].append(npi)
      index_cluster += 1
    
  return np_list


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
  
  # Menghitung N Precision dengan menggunakan tag & document
  all_np_top_list = get_all_np_top_list(tag_list, all_matrix_partition)
  all_np_list = get_all_np_list(tag_list, all_np_top_list)
  
  # Looping seluruh data di tag list
  for tag, cluster, nodes in tag_list:
    
    # nr = n_recall(matrix_w_original, cluster, nodes[0])
    # Menghitung np & ranki
    index_cluster = 0
    rank_i_list = []
    np_i_list = []
    for k in cluster:
      np = all_np_list[k-1][nodes[index_cluster + 1]]
      # np = n_precision(all_matrix_partition[k-1], nodes[index_cluster + 1])
      nr = n_recall(matrix_w_original, all_matrix_partition[k-1], nodes[0])
      ranki = rank(np, nr)
      index_cluster += 1
      np_i_list.append(np)
      rank_i_list.append(ranki)
    
    all_tag_list_with_rank.append([tag, cluster, nodes, rank_i_list, nr, np_i_list])
  
  return all_tag_list_with_rank

