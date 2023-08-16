import numpy as np

def top_k_accuracy(doc_list, dataframe_document_tag):
  """
    Mengkonversi data testing

  Args:
    doc_list: Daftar list dokumen
    dataframe_document_tag: Dataframe untuk dokumen dan tag
    
  Returns:
    success_list: list berapa tebakan yang benar
  """
  success_list = []
  for doc in doc_list:
    
    value = 0
    for tag in doc[-1]:
      if tag in dataframe_document_tag.columns:
        value += dataframe_document_tag.loc[doc[0][1], tag]
        if(value == 1):
          success_list.append(1)
          break
    
    if(value == 0):
      success_list.append(0)
  
  return success_list