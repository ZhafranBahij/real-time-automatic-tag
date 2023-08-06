import numpy as np


def word_count_in_matrix(document_list, all_matrix_partition):
  """
    Melakukan perhitungan berapa banyak kata dalam per dokumen

  Args:
    document_list: Daftar list dokumen
    all_matrix_partition: sebuah list yang berisi mengenai matriks-matriks yang telah dipartisi
    
  Returns:
    new_document_list: dokumen list dengan tambahan banyaknya kata dalam satu dokumen
    total_doc: menyimpan hasil berupa jumlah seluruh dokumen yg ada
    total_doc_per_cluster: banyaknya dokumen per klaster 
  """
  new_document_list = []
  total_doc = 0 # Menyimpan total doc yg ada
  total_doc_per_cluster = np.zeros(len(all_matrix_partition)) # Menyimpan total doc di masing-masing klaster

  # Looping sesuai banyaknya dokumen
  for title_and_id, cluster, nodes in document_list:
    index_cluster = 0
    word_count_list = []
    total_doc += 1
    
    # Looping klaster
    for k in cluster:
      # Mengambil row dari matriks partisi yang menandakan dokumen tersebut
      row_doc = all_matrix_partition[k-1][nodes[index_cluster + 1]]
      
      # Menghitung banyaknya kata
      word_count = sum(row_doc[int(len(row_doc)/2):])
      word_count_list.append(word_count)
      total_doc_per_cluster[k-1] += 1
    
    new_document_list.append([title_and_id, cluster, nodes, word_count_list])
    
  return new_document_list, total_doc, total_doc_per_cluster.tolist()

