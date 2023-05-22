import numpy as np

W = np.array([
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
[1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 0] ,
[1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0] ,
[1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 1] ,
[0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0] ,
[0 , 0 , 1 , 1 , 1 , 0 , 0 , 0 , 0] ,
[0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0] ,
])

# # Tag dengan doc
# # row = tag
# # col = doc
# A = [
#   [1, 1, 1],
#   [0, 0, 1]
# ]

# # doc dengan word
# # row = doc
# # col = word
# B = [
#   [1, 1, 1, 0],
#   [0, 1, 1, 0],
#   [0, 0, 0, 1],
# ]

def matrixABtoW(A, B):
    """
        Fungsi untuk menggabungkan matriks A dan B menjadi W

        Args:
            A: Matriks A dengan tag sebagai row dan document sebagai column
            B: Matriks B dengan document sebagai row dan word sebagai column

        Returns:
            W: Matriks gabungan antara A dengan 
    """
    AT = A.transpose()
    BT = B.transpose()

    # Menghitung banyaknya tag, dokumen, dan word
    tag_count, document_count = A.shape
    document_count, word_count = B.shape

    # Membuat matriks W dengan panjang dan lebar dari "tag + dokumen + word"
    all_count = tag_count + document_count + word_count
    W = np.zeros((all_count, all_count))

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

def getW():
    return W

# print(matrixABtoW(np.array(A), np.array(B)))