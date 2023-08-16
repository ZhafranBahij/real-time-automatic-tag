import numpy as np

def data_testing(doc_list, dataframe_document_word):
  """
    Mengkonversi data testing

  Args:
    doc_list: Daftar list dokumen
    dataframe_document_word: Dataframe untuk dokumen dan word
    
  Returns:
    new_doc_list: dokumen list dengan format baru
  """
  
  new_doc_list = []
  for doc in doc_list:
    # title_id, cluster, indexes, word_count, m_component
    new_doc_list.append([doc, [0], [0, 0], sum(dataframe_document_word.loc[int(doc[1])]), [0]])
    
  return new_doc_list