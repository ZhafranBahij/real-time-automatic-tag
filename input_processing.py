import data_from_database_full as dfd
import nltk
import numpy
import pandas
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Mengambil data dari database
dataset_document_word, dataset_tag_document = dfd.get_data()
# dataset_tag_document = dfd.get_document_and_tag()

# Tempat untuk menghitung banyaknya word dalam suatu dokumen
dataframe_document_word = []

def wordProcessing(content_article):
    """
    Fungsi untuk menghitung 'word' dalam suatu artikel
    """
    tokens = word_tokenize(re.sub('[^ 0-9a-z]+', ' ', content_article.lower())) # Menghilangkan tanda baca   
    english_stopwords = stopwords.words('english') # Menghilangkan stop words
    english_stopwords.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]

    # Menghitung banyaknya tokens
    freq = nltk.FreqDist(tokens_wo_stopwords)
    word_document_dictionary = {}

    # Menampung token tersebut dalam format dictionary
    for word, count in freq.most_common(999999):
        word_document_dictionary.update({word: count})
    
    return word_document_dictionary

def documentWordProcessing(content_article, title_article):
    """
    Fungsi untuk membuat dataframe antara 'document' dengan 'word'
    """
    word_document_dictionary = wordProcessing(content_article)
    datafr = pandas.DataFrame(word_document_dictionary,
        index=[title_article]
    )
    return datafr

for data in dataset_document_word:
    dataframe_document_word.append(documentWordProcessing(data[1], data[0]))
    
def document_tag_processing(dataset_tag_document):  
  title_before = dataset_tag_document[0][1]
  
  tag_dictionary = {}
  # Tempat untuk menghitung banyaknya tag dalam suatu dokumen
  dataframe_document_tag = []
  
  for data in dataset_tag_document:
    # Jika judul dokumen berbeda dengan row sebelumnya
    if title_before != data[1]:
      datafr = pandas.DataFrame(tag_dictionary,
          index=[title_before]
      )
      dataframe_document_tag.append(datafr)
      title_before = data[1]
      tag_dictionary.clear()
    
    tag_dictionary.update({data[0]: 1})

  datafr = pandas.DataFrame(tag_dictionary,
      index=[title_before]
  )
  dataframe_document_tag.append(datafr)
  tag_dictionary.clear()
  
  return dataframe_document_tag
    
dataframe_document_tag_join = pandas.concat(document_tag_processing(dataset_tag_document))
dataframe_document_tag_join = dataframe_document_tag_join.fillna(0)
print(dataframe_document_tag_join)

dataframe_document_word_join = pandas.concat(dataframe_document_word)
dataframe_document_word_join = dataframe_document_word_join.fillna(0)
print(dataframe_document_word_join)

def matrixABtoW(A, B):
    """
    Fungsi untuk memasukkan matriks A, A transpose, B, dan B transpose ke dalam matriks W
    """
    AT = A.transpose()
    BT = B.transpose()

    tag_count, document_count = A.shape
    document_count, word_count = B.shape

    all_count = tag_count + document_count + word_count
    W = numpy.zeros((all_count, all_count))

    # Menempelkan matriks A ke W
    for i in range(tag_count):
        W[i][tag_count:-word_count] = A[i]

    # Menempelkan matriks B Transpose ke W
    for i in range(1, word_count+1):
        W[-i][tag_count:-word_count]= BT[-i]

    # Menempelkan matriks A Transpose ke W
    for i in range(document_count):
        W[tag_count+i][0:tag_count] = AT[i]
        
    # Menempelkan matriks B ke W
    for i in range(document_count):
        W[tag_count+i][-word_count:] = B[i]
    
    return W

# print(dataframe_document_word_join.to_numpy())
# print(dataframe_document_tag_join.to_numpy())

matrix_w = matrixABtoW(dataframe_document_tag_join.to_numpy().transpose(), dataframe_document_word_join.to_numpy())
print(matrix_w)

