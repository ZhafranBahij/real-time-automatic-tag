def assign_label_cluster (title_id_document, all_tag_list, all_word_list, all_cluster): 
  
  # Memasukkan label klaster ke dalam tag
  index = 0 # Index row matrix awal
  all_tag_list_with_cluster = []
  
  # Looping seluruh isi all_tag_list
  for tag in all_tag_list:
    tag_cluster = []
    index_cluster = 0 # index klaster
    
    # Looping selurung klaster
    for cluster in all_cluster:
      if index in cluster:
        tag_cluster.append(index_cluster + 1)
      index_cluster+=1
      
    index += 1
    all_tag_list_with_cluster.append([tag, tag_cluster])

  # Memasukkan label klaster ke dalam document
  all_title_id_document_with_cluster = []
  
  # Looping seluruh isi title_id_document
  for title, id in title_id_document:
    document_cluster = []
    index_cluster = 0

    # Looping isi klaster
    for cluster in all_cluster:
      # Jika index tersebut ada di suatu klaster
      if index in cluster:
        document_cluster.append(index_cluster + 1)
      index_cluster+=1

    index += 1
    all_title_id_document_with_cluster.append([[title, id], document_cluster])

  # Memasukkan label klaster ke dalam word
  all_word_list_with_cluster = []
  
  # Looping seluruh word di all_word_list
  for word in all_word_list:
    word_cluster = []
    index_cluster = 0
    
    # Looping isi klaster
    for cluster in all_cluster:
      if index in cluster:
        word_cluster.append(index_cluster + 1)
      index_cluster+=1

    index += 1
    all_word_list_with_cluster.append([word, word_cluster])
  
  return all_tag_list_with_cluster, all_title_id_document_with_cluster, all_word_list_with_cluster