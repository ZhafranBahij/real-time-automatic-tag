import pandas

def tag_recommendation(all_tag_list_with_rank, all_cluster, all_cluster_word):

  """
  Melakukan rekomendasi tag terhadap dokumen baru
  """
  
  p_cluster = 1/len(all_cluster) # P(C=k)
  p_document = [1/(len(k)+1) for k in all_cluster_word] # list P(D = dt) || Setiap klaster berbeda valuenya
  p_dt_ck = 0.25
  
  R_Ti_dt = [] # Tampungan untuk nilai rank akhir
  all_tag_name = [] # Tampungan untuk nama tag

  for tag, cluster, nodes, rank, nr, np_i in all_tag_list_with_rank:
    index_cluster = 0
    
    for k in cluster:
      probability = p_dt_ck * p_cluster / p_document[index_cluster] # Hitung P(C=k|D=dt)
      rti = rank[index_cluster] * probability # Hitung R(Ti, dt)
      R_Ti_dt.append(rti)
      all_tag_name.append(tag)
      index_cluster += 1

  # Buat dataframe dgn kolom tag & rank akhir lalu urutkan
  dff = pandas.DataFrame([all_tag_name, R_Ti_dt], ["Tag", "Value"])
  dffT = dff.T.sort_values(by=['Value'], ascending=False)

  # Ambil 10 tag dgn rank akhir terbesar
  big_10_rank = [tag for tag in dffT.head(10)["Tag"]]
  
  return big_10_rank