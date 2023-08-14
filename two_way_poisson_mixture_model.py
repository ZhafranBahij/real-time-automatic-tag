import numpy as np
import pandas as pd

def set_m_component_to_document(doc_list, M, K):
    """
    Melabelkan dokumen dengan m komponen

    Args:
        doc_list: daftar dari dokumen
        M: banyaknya m komponen
        K: banyaknya klaster

    Returns:
        new_doc_list: prior probability
        total_doc_in_component: Banyaknya dokumen dalam suatu komponen
    """
    new_doc_list = []
    total_doc_in_component = np.zeros(M)
    index_component = np.zeros(K)
    
    for title_id, cluster, indexes, word_count in doc_list:
        m_list = []
        
        # cluster = klaster dari dokumen
        for k in cluster:
            # Mendapatkan m ke berapa
            # (M / K)*(k-1) + 1 berguna untuk menentukan titik mulai-nya komponen berdasarkan K dan M
            # index_component[k-1] % (M / K) berguna untuk membirkan value m secara bergantian setiap klaster
            m_component = int((int(M / K)*(k-1) + 1) + (index_component[k-1] % int(M / K)))
            m_list.append(m_component) 
            total_doc_in_component[m_component - 1] += 1
            index_component[k-1] += 1

        # Membuat list dokumen terbaru
        new_doc_list.append([title_id, cluster, indexes, word_count, m_list])
    
    return new_doc_list, total_doc_in_component

def set_word_count_in_every_m(doc_list, word_list, M, K, matrix_document_word):
    """
    Menghitung banyaknya masing-masing word di setiap komponen

    Args:
        doc_list: daftar dari dokumen
        word_list: daftar word
        M: banyaknya m komponen
        K: banyaknya klaster
        matrix_document_word: matrix relasi antara dokumen sbg row dgn word sbg column

    Returns:
        new_doc_list: prior probability
        total_doc_in_component: Banyaknya dokumen dalam suatu komponen
    """
    new_word_list = []
    row, col = matrix_document_word.shape
    total_every_word_in_component = np.zeros((M, col)) # Inisiasi word dalam component
    
    index = 0
    # Proses menghitung banyaknya word di dalam suatu komponen doc
    for title_id, cluster, indexes, word_count, m_component in doc_list:
        for m in m_component:
            total_every_word_in_component[m-1] += matrix_document_word[index]
        index += 1
    

    index = 0
    for word, cluster, indexes, word_count in word_list:
        # Memindahkan total_every_word_in_component ke dalam word_count sesuai word-nya
        word_count = total_every_word_in_component[..., index]
        index += 1
        
        new_word_list.append([word, cluster, indexes, word_count.tolist()])

    return new_word_list

def first_prior_probability(total_doc, total_doc_in_component):
    """
    Menghitung pi_m dengan cara mencari prior probability setiap m
    Dengan asumsi banyaknya M adalah banyaknya K

    Args:
      total_doc: Keseluruhan dokumen dari dataset yang diberikan
      total_doc_in_component: Keselurhan dokumen dalam satu M

    Returns:
        pi_m: prior probability
    """
    
    # Hitung nilai pi_m di setiap M
    pi_m = np.array(total_doc_in_component) / total_doc
    return pi_m

def lambda_m_j_list(word_list, total_doc_in_component):
    """
    Menghitung nilai lambda untuk setiap kata

    Args:
      word_list: list seluruh word yg ada di dataset
      total_doc_in_component: total dokumen dalam 1 komponen

    Returns:
        pi_m: prior probability
    """
    
    new_word_list = [] # word list baru
    
    # Looping word list untuk mencari lambda_m_j
    for word, cluster, indexes, word_count in word_list:
        lambda_m_j = []
        index_m = 0
        for tdic in total_doc_in_component:
            lambda_m_j.append(word_count[index_m] / tdic)
            index_m += 1 
            # lambda_m_j.append(1)
        
        new_word_list.append([word, cluster, indexes, word_count, lambda_m_j])
    
    return new_word_list

def probability_mass_function(d_ij, lambda_mij):
    """
    Menghitung probability mass function pada suatu word

    Args:
        d_ij: banyaknya word j dalam dokumen i
        total_doc_in_cluster: lambda dari word j

    Returns:
        teta: probability mass function
    """
    
    teta = np.exp(-lambda_mij) * np.power(lambda_mij, d_ij) / np.prod(np.arange(1, d_ij+1))
    return teta

def probability(doc_list, pi_m, word_list, dataframe_document_word, M):
    """
    Menghitung P(D = d|C = k) untuk setiap dokumen
    
    Args:
        doc_list: list dari suatu dokumen
        pi_m: nilai π_m (prior probability) setiap komponen
        word_list: list dari suatu word
        dataframe_document_word: Dataframe relasi antara doc dgn word

    Returns:
        new_doc_list: list doc dgn tambahan probability
    """
    new_doc_list = []
    
    for title_id, cluster, indexes, word_count, m_component in doc_list:
        probability = [] # Nilai probability yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        prod_teta_list = np.ones(M)
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            if(word_value < 1):
                i += 1
                continue
            
            for m in m_component:
                # Memasukkan probability mass function dgn
                # word_value = banyaknya word dari doc ini
                # word_list[i][4][m-1] = lambda_mj dari word tersebut
                prod_teta_list[m-1] *= probability_mass_function(word_value, word_list[i][4][m-1])
            i+=1
        
        # Menghitung probability
        for m in m_component:
            # prod_teta_list = np.prod(teta_list[m-1])
            probability.append(pi_m[m-1] * prod_teta_list[m-1])
            
        new_doc_list.append([title_id, cluster, indexes, word_count, m_component, sum(probability)])
        
    return new_doc_list

def p_im_list(doc_list, pi_m, word_list, dataframe_document_word, M):
    """
        Memproses p_im

        Args:
            doc_list: daftar dokumen
            pi_m: prior probability dari komponen m dgn asumsi banyaknya K = banyaknya M
            word_list: list dari word
            dataframe_document_word: dataframe dengan document sebagai row dan word sebagai column
            
        Returns:
            new_doc_list: list doc terbaru 
    """
    
    new_doc_list = []
    
    # Looping doc_list dgn:
    # title_id: judul dan id dari doc
    # cluster: klaster dari dokumen
    # indexes: posisi row index pada matrix w dan matrix w partition
    # word_count: banyaknya jumlah word dalam dokumen
    for title_id, cluster, indexes, word_count, m_component, probability in doc_list:
        p_im = [] # Nilai p_im yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        prod_teta_list = np.ones(M)
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            if(word_value < 1):
                i += 1
                continue
            
            for m in m_component:
                # Memasukkan probability mass function dgn
                # word_value = banyaknya word dari doc ini
                # word_list[i][4][m-1] = lambda_mj dari word tersebut
                prod_teta_list[m-1] *= probability_mass_function(word_value, word_list[i][4][m-1])
            i+=1
        
        # Menghitung p_im
        for m in m_component:
            p_im.append(pi_m[m-1] * prod_teta_list[m-1])
            
        new_doc_list.append([title_id, cluster, indexes, word_count, m_component, p_im, probability])
        
    return new_doc_list

def pi_m_with_t(doc_list, M):
    """
        Mencari nilai p_im jika pencarian p_im lebih dari 1 turn

        Args:
            doc_list: daftar dokumen
            M: banyaknya komponen
        Returns:
            pi_m_list: Daftar pi_m terbaru
            sum_p_im_list: sum dari p_im pada turn saat ini di setiap komponen
    """
    
    pi_m_list = np.zeros(M)
    sum_p_im_list = np.zeros(M)

    # Mencari nilai sum(p_im)
    for title_id, cluster, indexes, word_count, m_component, p_im, probability in doc_list:
        index_m = 0
        for m in m_component:
            sum_p_im_list[m-1] += p_im[index_m]
            index_m += 1
    
    # Mencari nilai pi_m
    index = 0
    for value in sum_p_im_list:
        pi_m_list[index] = value/sum(sum_p_im_list)
        index += 1
    
    return pi_m_list, sum_p_im_list

def lambda_mt(word_list, sum_p_im_list, doc_list, M):
    """
        Mencari nilai lambda jika pencarian lambda lebih dari 1 turn

        Args:
            word_list: daftar word
            sum_p_im_list: sum dari p_im pada turn saat ini di setiap komponen
        Returns:
            new_word_list: daftar word terbaru
    """
    
    new_word_list = [] # word list baru
    top_lambda_mt_list = np.zeros(M) # Untuk perhitungan pada persamaan lambda_mt bagian atas 
    
    # Looping setiap dokumen
    for title_id, cluster, indexes, word_count, m_component, p_im, probability in doc_list:
        
        index_m = 0
        for m in m_component:
            top_lambda_mt_list[m-1] += p_im[index_m] * word_count
            index_m += 1
    
    # Looping word list untuk mencari lambda_m_j
    for word, cluster, indexes, word_count, lambda_m_j in word_list:
        
        lambda_m_j_temp = []
        
        # Kalkulasi nilai lambda berdasarkan word dan klasternya
        for m in range(1, M+1):
            bottom_lambda_mt = word_count[m-1] * sum_p_im_list[m-1]
            
            # Jika nilai bottom_lambda_mt terbaru bernilai 0 dan mencegah lambda_m_j_temp bernilai inf
            if bottom_lambda_mt == 0:
                lambda_m_j_temp.append(0)
            else:
                lambda_m_j_temp.append(top_lambda_mt_list[m-1] / word_count[m-1] * sum_p_im_list[m-1])
            
        
        new_word_list.append([word, cluster, indexes, word_count, lambda_m_j_temp])
    
    return new_word_list

def p_im_list_t_more_than_1(doc_list, pi_m, word_list, dataframe_document_word):
    """
        Mencari nilai p_im jika pencarian p_im lebih dari 1 turn

        Args:
            doc_list: daftar dokumen
            pi_m: prior probability dari komponen m
            word_list: list dari word
            dataframe_document_word: dataframe dengan document sebagai row dan word sebagai column
        Returns:
    """
    
    new_doc_list = []
    
    # Looping doc_list dgn:
    # title_id: judul dan id dari doc
    # cluster: klaster dari dokumen
    # indexes: posisi row index pada matrix w dan matrix w partition
    # word_count: banyaknya jumlah word dalam dokumen
    # p_im: Nilai dari p_im
    for title_id, cluster, indexes, word_count, m_component, p_im, probability in doc_list:
        p_im = [] # Nilai p_im yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        teta_list = []
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            teta_list.append(probability_mass_function(word_value, word_list[i][4][0]))
        
        # Menghitung p_im
        for m in m_component:
            prod_teta_list = np.prod(teta_list)
            p_im.append(pi_m[m-1] * prod_teta_list)
            
        new_doc_list.append([title_id, cluster, indexes, word_count, m_component, p_im, probability])
        
    return new_doc_list

def get_log_likelihood(doc_list, new_doc_list):
    log_likelihood = 0
        
    # Looping dokumen
    for i in range(0, len(doc_list)):
        
        # Looping sebanyak label m yg ada di dokumen
        for m in range(0, len(doc_list[i][4])):  
            log_p_im = np.log(new_doc_list[i][5][m])
            log_likelihood += doc_list[i][5][m] * log_p_im
        
    return log_likelihood