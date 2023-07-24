def assign_label_for_tag(all_tag_list, all_cluster, index):
  all_tag_list_with_cluster = []
  
  # Looping seluruh isi all_tag_list
  for tag in all_tag_list:
    tag_cluster = []
    tag_index_in_matrix = [] # Kumpulan index dalam satu tag di dalam setiap matrix yg ditempati tag tersebut
    index_cluster = 0 # index klaster
    
    tag_index_in_matrix.append(index)
    # Looping selurung klaster
    for cluster in all_cluster:
      if index in cluster:
        tag_cluster.append(index_cluster + 1)
        tag_index_in_matrix.append(cluster.index(index))
      index_cluster+=1
      
    index += 1
    all_tag_list_with_cluster.append([tag, tag_cluster, tag_index_in_matrix])
  
  return index, all_tag_list_with_cluster
  
def assign_label_for_document(title_id_document, all_cluster, index):
  # Memasukkan label klaster ke dalam document
  all_title_id_document_with_cluster = []
  
  # Looping seluruh isi title_id_document
  for title, id in title_id_document:
    document_cluster = []
    document_index_in_matrix = [] # Kumpulan index dalam satu document di dalam setiap matrix yg ditempati document tersebut
    index_cluster = 0

    document_index_in_matrix.append(index)
    # Looping isi klaster
    for cluster in all_cluster:
      # Jika index tersebut ada di suatu klaster
      if index in cluster:
        document_cluster.append(index_cluster + 1)
        document_index_in_matrix.append(cluster.index(index))
      index_cluster+=1

    index += 1
    all_title_id_document_with_cluster.append([[title, id], document_cluster, document_index_in_matrix])

  return index, all_title_id_document_with_cluster

def assign_label_for_word(all_word_list, all_cluster, index):
  # Memasukkan label klaster ke dalam word
  all_word_list_with_cluster = []
  
  # Looping seluruh word di all_word_list
  for word in all_word_list:
    word_cluster = []
    word_index_in_matrix = [] # Kumpulan index dalam satu word di dalam setiap matrix yg ditempati word tersebut
    index_cluster = 0
    
    word_index_in_matrix.append(index)
    # Looping isi klaster
    for cluster in all_cluster:
      if index in cluster:
        word_cluster.append(index_cluster + 1)
        word_index_in_matrix.append(cluster.index(index))
      index_cluster+=1

    index += 1
    all_word_list_with_cluster.append([word, word_cluster, word_index_in_matrix])
  
  return index, all_word_list_with_cluster

def assign_label_cluster(title_id_document, all_tag_list, all_word_list, all_cluster): 
  
  # Memasukkan label klaster ke dalam tag
  index = 0 # Index row matrix awal
  index, all_tag_list_with_cluster = assign_label_for_tag(all_tag_list, all_cluster, index)
  index, all_title_id_document_with_cluster = assign_label_for_document(title_id_document, all_cluster, index)
  index, all_word_list_with_cluster = assign_label_for_word(all_word_list, all_cluster, index)

  return all_tag_list_with_cluster, all_title_id_document_with_cluster, all_word_list_with_cluster