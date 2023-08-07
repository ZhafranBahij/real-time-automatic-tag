import numpy as np


def word_count_in_matrix(document_list, all_matrix_partition, dataframe_document_word):
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
    # index_cluster = 0
    total_doc += 1
    
    # Mengambil row dari matriks partisi yang menandakan dokumen tersebut
    # row_doc = matrix_origin[nodes[0]]
    
    # Menghitung banyaknya kata
    word_count = sum(dataframe_document_word.loc[title_and_id[1]])

    # Looping klaster
    for k in cluster:
      total_doc_per_cluster[k-1] += 1
    
    new_document_list.append([title_and_id, cluster, nodes, word_count])
    
  return new_document_list, total_doc, total_doc_per_cluster.tolist()

