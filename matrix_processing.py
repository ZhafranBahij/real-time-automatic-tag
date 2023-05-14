import numpy

W = [
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
[1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0] ,
[1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0] ,
[1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1] ,
[0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
]

# Tag dengan doc
# row = tag
# col = doc
A = [
  [1, 1, 1],
  [0, 0, 1]
]

# doc dengan word
# row = doc
# col = word
B = [
  [1, 1, 1, 0],
  [0, 1, 1, 0],
  [0, 0, 0, 1],
]

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
  
print(matrixABtoW(numpy.array(A), numpy.array(B)))