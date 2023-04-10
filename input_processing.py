import data_from_database as dfd
import nltk
import numpy
import pandas

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Mengambil data dari database
dataset = dfd.get_document_and_word()

# Tempat untuk menghitung banyaknya word dalam suatu dokumen
dataframe_document_word = []

# Tempat untuk menghitung banyaknya tag dalam suatu dokumen
dataframe_document_tag = []

def wordProcessing(content_article):
    """
    Fungsi untuk menghitung 'word' dalam suatu artikel
    """
    tokens = word_tokenize(content_article.lower())
    english_stopwords = stopwords.words('english')
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]

    freq = nltk.FreqDist(tokens_wo_stopwords)
    word_document_dictionary = {}

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

def tagProcessing(tag_article):
    """
    Fungsi untuk menghitung 'tag' dalam suatu artikel 
    """
    tag_article = tag_article.replace('\t', '')
    tokens = nltk.regexp_tokenize(tag_article.lower(), '\n', gaps=True)

    freq = nltk.FreqDist(tokens)
    tag_document_dictionary = {}

    for tag, count in freq.most_common(999999):
        tag_document_dictionary.update({tag: count})
    
    return tag_document_dictionary

def documentTagProcessing(tag_article, title_article):
    tag_document_dictionary = tagProcessing(tag_article)
    datafr = pandas.DataFrame(tag_document_dictionary,
        index=[title_article]
    )
    return datafr

for data in dataset:
    dataframe_document_word.append(documentWordProcessing(data[2], data[0]))
    dataframe_document_tag.append(documentTagProcessing(data[1], data[0]))

dataframe_document_word_join = pandas.concat(dataframe_document_word)
dataframe_document_word_join = dataframe_document_word_join.fillna(0)
dataframe_document_tag_join = pandas.concat(dataframe_document_tag)
dataframe_document_tag_join = dataframe_document_tag_join.fillna(0)

# def dataframeToMatrix(document_word, document_tag):

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