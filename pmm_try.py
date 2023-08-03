import numpy as np
import pandas as pd

# def poisson_pdf(x, lambd):
#     return np.exp(-lambd) * np.power(lambd, x) / np.math.factorial(x)

# def log_likelihood(data, weights, lambdas):
#     n_components = len(weights)
#     log_likelihoods = np.zeros(len(data))
#     for i in range(n_components):
#         log_likelihoods += weights[i] * poisson_pdf(data, lambdas[i])
#     return np.sum(np.log(log_likelihoods))

# def expectation_maximization(data, n_components, max_iter=100, tol=1e-6):
#     n_samples = len(data)
#     weights = np.ones(n_components) / n_components
#     lambdas = np.random.rand(n_components)

#     for _ in range(max_iter):
#         # E-step
#         responsibilities = np.zeros((n_samples, n_components))
#         for i in range(n_components):
#             responsibilities[:, i] = weights[i] * poisson_pdf(data, lambdas[i])

#         responsibilities /= responsibilities.sum(axis=1, keepdims=True)

#         # M-step
#         new_lambdas = np.zeros(n_components)
#         for i in range(n_components):
#             new_lambdas[i] = np.sum(responsibilities[:, i] * data) / np.sum(responsibilities[:, i])

#         # Check for convergence
#         if np.all(np.abs(new_lambdas - lambdas) < tol):
#             break

#         lambdas = new_lambdas

#     return weights, lambdas

# # Example usage
# if __name__ == "__main__":
#     np.random.seed(0)
#     data = np.random.poisson(lam=5, size=1000)  # Simulated Poisson data with mean 5

#     n_components = 2
#     weights, lambdas = expectation_maximization(data, n_components)

#     print("Estimated Weights:", weights)
#     print("Estimated Lambdas:", lambdas)
#     print("Log Likelihood:", log_likelihood(data, weights, lambdas))

# def first_prior_probability(document_list, M):
#     pi_m = np.zeros(M)
#     total_document_in_m = np.zeros(M)
    
#     # Untuk sementara, anggap m = k
#     for title_and_id, cluster, nodes, word_count in document_list:
#         for k in cluster:
#             total_document_in_m[k-1] += 1
    
#     # Hitung prior probability
#     index = 0
#     sum_total_document_in_m = sum(total_document_in_m)
#     for tdim in total_document_in_m:
#         pi_m[index] = tdim / sum_total_document_in_m
#         index += 1
        
#     return pi_m

def first_prior_probability(total_word, total_word_in_cluster):
    pi_m = np.zeros(len(total_word_in_cluster))
    
    index = 0
    for twic in total_word_in_cluster:
        pi_m[index] = twic/total_word
        index += 1
        
    return pi_m

def probability_mass_function(lambda_lm, d_kl):
    value_top = np.exp(-lambda_lm) * np.power(lambda_lm, d_kl)
    value_bottom = np.math.factorial(d_kl)
    value = value_top / value_bottom
    return value


def pi(m_value, p_im):
    pi_top = sum(p_im[m_value-1]) #sum seluruh p_im sesuai M ke berapa
    pi_bottom = sum(sum(p_im)) #sum seluruh p_im yg ada
    pi = pi_top / pi_bottom
    return pi

def lambda_m(m_value, p_im, count_word_in_doc):
    # sum bagian atas dari persamaan 15
    lambda_top = 0
    for i in p_im[m_value-1]:
        lambda_top += i * count_word_in_doc

    # sum bagian bawah dari persamaan 15
    lambda_bottom = sum(count_word_in_doc) * sum(p_im[m_value-1])
    
    #
    lambda_m = lambda_top / lambda_bottom
    
    return lambda_m

# temp = probability_mass_function(100, 100)
# print(temp)