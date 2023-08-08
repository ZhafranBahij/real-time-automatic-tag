import numpy as np
import pandas as pd

def set_m_component_to_document(doc_list, M, K):
    
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
    
        new_doc_list.append([title_id, cluster, indexes, word_count, m_list])
    
    return new_doc_list, total_doc_in_component

def probability(doc_list, pi_m, word_list, dataframe_document_word):
    
    new_doc_list = []
    
    for title_id, cluster, indexes, word_count, m_component in doc_list:
        probability = [] # Nilai probability yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        teta_list = []
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            if(word_value < 1):
                i += 1
                continue
            teta_list.append(probability_mass_function(word_value, word_list[i][4][0]))
            i+=1
        
        # Menghitung probability
        for m in m_component:
            prod_teta_list = np.prod(teta_list)
            probability.append(pi_m[m-1] * prod_teta_list)
            
        new_doc_list.append([title_id, cluster, indexes, word_count, m_component, sum(probability)])
        
    return new_doc_list

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
        for tdic in total_doc_in_component:
            # lambda_m_j.append(word_count[0] / tdic) 
            lambda_m_j.append(1)
        
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

def p_im_list(doc_list, pi_m, word_list, dataframe_document_word):
    """
        Memproses p_im

        Args:
            doc_list: daftar dokumen
            pi_m: prior probability dari komponen m dgn asumsi banyaknya K = banyaknya M
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
    for title_id, cluster, indexes, word_count, m_component in doc_list:
        p_im = [] # Nilai p_im yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        teta_list = []
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            teta_list.append(probability_mass_function(word_value, word_list[i][4][0]))
            i+=1
        
        # Menghitung p_im
        for m in m_component:
            prod_teta_list = np.prod(teta_list)
            p_im.append(pi_m[m-1] * prod_teta_list)
            
        new_doc_list.append([title_id, cluster, indexes, word_count, m_component, p_im])
        
    return new_doc_list

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
    for title_id, cluster, indexes, word_count, p_im in doc_list:
        p_im = [] # Nilai p_im yg akan distore di doc list baru
        
        # Menghitung teta di setiap kata di dalam 1 dokumen
        teta_list = []
        i = 0
        for word_value in dataframe_document_word.loc[title_id[1]]:
            teta_list.append(probability_mass_function(word_value, word_list[i][4][0]))
        
        # Menghitung p_im
        for k in cluster:
            prod_teta_list = np.prod(teta_list)
            p_im.append(pi_m[k-1] * prod_teta_list)
            
        new_doc_list.append([title_id, cluster, indexes, word_count, p_im])
        
    return new_doc_list


def pi_m_with_t(doc_list, m = 2):
    """
        Mencari nilai p_im jika pencarian p_im lebih dari 1 turn

        Args:
            doc_list: daftar dokumen
            m: banyaknya komponen
        Returns:
            pi_m_list: Daftar pi_m terbaru
            sum_p_im_list: sum dari p_im pada turn saat ini di setiap komponen
    """
    
    pi_m_list = np.zeros(m)
    sum_p_im_list = np.zeros(m)

    # Mencari nilai sum(p_im)
    for title_id, cluster, indexes, word_count, m_component, p_im in doc_list:
        for m in m_component:
            sum_p_im_list[m-1] += p_im
    
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
    top_lambda_mt_list = np.zeros(M) # Angka 2 tergantung m-nya
    
    # Looping setiap dokumen
    for title_id, cluster, indexes, word_count, m_component, p_im in doc_list:
        
        index_m = 0
        for m in m_component:
            top_lambda_mt_list[m-1] += p_im[index_m] * word_count
            index_m += 1
    
    # Looping word list untuk mencari lambda_m_j
    for word, cluster, indexes, word_count, lambda_m_j in word_list:
        
        lambda_m_j_temp = []
        index_cluster = 0
        
        # Kalkulasi nilai lambda berdasarkan word dan klasternya
        for m in m_component:
            lambda_m_j_temp.append(top_lambda_mt_list[m-1] / word_count[index_cluster + 1] * sum_p_im_list[m-1])
            index_cluster += 1
        
        new_word_list.append([word, cluster, indexes, word_count, lambda_m_j_temp])
    
    return new_word_list

def get_L(doc_list):
    L = 0
    for title_id, cluster, indexes, word_count, p_im in doc_list:
        log_p_im = np.log(p_im)
        L += p_im * log_p_im
        
    return L