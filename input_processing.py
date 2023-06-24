import nltk
import numpy
import pandas
import re
import datetime

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def wordProcessing(content_article):
    """
    Fungsi untuk menghitung banyaknya word dalam suatu artikel

    Args:
      content_article: Isi dari artikel
      
    Sumber code memfilter stopwords: 
      https://medium.com/analytics-vidhya/removing-stop-words-with-nltk-library-in-python-f33f53556cc1
    
    Sumber library untuk menghitung kata:
      https://www.nltk.org/book/ch01.html
      
    Returns:
        word_document_dictionary: berupa dictionary untuk menghitung banyaknya 
                                  dan beragamnya word dalam suatu document
    """
    tokens = word_tokenize(re.sub('[^ 0-9a-z]+', ' ', content_article.lower())) # Menghilangkan tanda baca   
    english_stopwords = stopwords.words('english') # Menampilkan daftar stopwords
    english_stopwords.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']) #menambahkan stopwords
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords] 

    # Menghitung banyaknya word
    freq = nltk.FreqDist(tokens_wo_stopwords)
    word_document_dictionary = {}

    # Menampung word tersebut dalam format dictionary
    for word, count in freq.most_common(1000):
        word_document_dictionary.update({word: count})
        
    # print("Dictionary : ", word_document_dictionary)
    return word_document_dictionary

def documentWordProcessing(content_article, id_article):
    """
    Fungsi untuk membuat dataframe antara document(title) dengan word

    Args:
      content_article: Isi dari artikel
      id_article: ID dari artikel

    Returns:
        word_document: word_document dalam bentuk dataframe
    """
    word_document_dictionary = wordProcessing(content_article)
    word_document = pandas.DataFrame(word_document_dictionary,
        index=[id_article]
    )
    return word_document
  
def document_processing(dataset_document):
  """
  Memproses dataset yang masuk, lalu mengolahnya menjadi kumpulan dataframe antara tag dgn dokumen
  dan dokumen dgn word

  Args:
    dataset_document: data-data yg diambil dari database dengan isi "tag, id_article, content_article"
    
  Returns:
    matrix_tag_document: matriks antara tag dgn document
    matrix_document_word: matriks antara document dgn word
    title_id_document: relasi antara title dan id dari suatu artikel
  """
  now = datetime.datetime.now()
  print('Start Document Processing :', now)
  id_before = dataset_document[0][1]
  document_word = []
  document_word.append(documentWordProcessing(dataset_document[0][2], dataset_document[0][1]))
  
  title_id_document = []
  title_id_document.append((dataset_document[0][3].replace('| The Hill', ''), dataset_document[0][1]))
  
  
  tag_dictionary = {}
  # Tempat untuk menghitung banyaknya tag dalam suatu dokumen
  document_tag = []
  
  now = datetime.datetime.now()
  print('Start Looping :', now)
  for data in dataset_document:
    
    # Jika judul data berbeda dengan id_before
    if id_before != data[1]:
      
      # Menampung tag-tag yg telah didapat di tag_dictionary ke document_tag
      datafr = pandas.DataFrame(tag_dictionary,
          index=[id_before]
      )
      document_tag.append(datafr)
      tag_dictionary.clear()
      
      # Melakukan proses untuk menghitung banyaknya kata dalam suatu dokumen
      document_word.append(documentWordProcessing(data[2], data[1]))
      title_id_document.append((data[3].replace('| The Hill', ''), data[1]))
      id_before = data[1]
    
    #tag yg didapat akan dimasukkan ke tag_dictionary
    tag_dictionary.update({data[0]: 1})

  datafr = pandas.DataFrame(tag_dictionary,
      index=[id_before]
  )
  document_tag.append(datafr)
  tag_dictionary.clear()
  now = datetime.datetime.now()
  print('End Looping :', now)
  
  # document_tag, document_word = document_processing(dataset_document)

  now = datetime.datetime.now()
  print('Start Concat Doctag :', now)
  document_tag = pandas.concat(document_tag)
  document_tag = document_tag.fillna(0)
  # print(document_tag)

  now = datetime.datetime.now()
  print('Start Concat Docword :', now)
  document_word = pandas.concat(document_word)
  now = datetime.datetime.now()
  print('Start fillna Docword :', now)
  document_word = document_word.fillna(0)
  # print(document_word)  
  
  now = datetime.datetime.now()
  print('Start Create matrix :', now)
  matrix_tag_document = document_tag.to_numpy().transpose()
  matrix_document_word = document_word.to_numpy()
  
  now = datetime.datetime.now()
  print('End Document Processing :', now)
  return matrix_tag_document, matrix_document_word, title_id_document

