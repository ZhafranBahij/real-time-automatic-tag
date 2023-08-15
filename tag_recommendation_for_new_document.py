import pandas

def tag_recommendation(all_tag_list_with_rank, all_cluster, total_doc_in_cluster, p_dt_ck):

  """
  Melakukan rekomendasi tag terhadap dokumen baru
  """
  
  p_cluster = 1/len(all_cluster) # P(C=k)
  p_document = [1/tdic for tdic in total_doc_in_cluster] # list P(D = dt) || Setiap klaster berbeda valuenya
  # p_dt_ck = 0.25
  
  R_Ti_dt = [] # Tampungan untuk nilai rank akhir
  all_tag_name = [] # Tampungan untuk nama tag

  for tag, cluster, nodes, rank, nr, np_i in all_tag_list_with_rank:
    index_cluster = 0
    
    for k in cluster:
      probability = p_dt_ck * p_cluster / p_document[k-1] # Hitung P(C=k|D=dt)
      rti = rank[index_cluster] * probability # Hitung R(Ti, dt)
      R_Ti_dt.append(rti)
      all_tag_name.append(tag)
      index_cluster += 1

  # Buat dataframe dgn kolom tag & rank akhir lalu urutkan
  dff = pandas.DataFrame([all_tag_name, R_Ti_dt], ["Tag", "Value"])
  dffT = dff.T.sort_values(by=['Value'], ascending=False)

  # Ambil 6 tag dgn rank akhir terbesar
  big_rank = [tag for tag in dffT.head(6)["Tag"]]
  
  return big_rank

def tag_recommendation_mass(doc_list, all_tag_list_with_rank, all_cluster, total_doc_in_cluster):
  
  new_doc_list= []
  
  index = 0
  for doc in doc_list:
    tag_recommend = tag_recommendation(all_tag_list_with_rank, all_cluster, total_doc_in_cluster, doc[-1])
    doc_list[index].append(tag_recommend)
    index += 1
  
  return doc_list
  