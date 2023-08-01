# import numpy as np

# def poisson_likelihood(data, lambdas):
#     return np.prod(np.exp(-lambdas) * (lambdas ** data) / np.math.factorial(data))

# def expectation_step(data, lambdas, pi):
#     num_clusters = len(lambdas)
#     num_samples = data.shape[0]
#     responsibilities = np.zeros((num_samples, num_clusters))

#     for i in range(num_samples):
#         for k in range(num_clusters):
#             responsibilities[i, k] = pi[k] * poisson_likelihood(data[i], lambdas[k])

#         responsibilities[i] /= np.sum(responsibilities[i])

#     return responsibilities

# def maximization_step(data, responsibilities):
#     num_clusters = responsibilities.shape[1]
#     num_samples = data.shape[0]
#     num_features = data.shape[1]

#     lambdas = np.zeros((num_clusters, num_features))
#     pi = np.zeros(num_clusters)

#     for k in range(num_clusters):
#         Nk = np.sum(responsibilities[:, k])
#         pi[k] = Nk / num_samples

#         for f in range(num_features):
#             lambdas[k, f] = np.sum(responsibilities[:, k] * data[:, f]) / Nk

#     return lambdas, pi

# def two_way_poisson_mixture_model(data, num_clusters, max_iterations=100, tolerance=1e-6):
#     num_samples = data.shape[0]
#     num_features = data.shape[1]

#     # Random initialization
#     np.random.seed(42)
#     lambdas = np.random.rand(num_clusters, num_features)
#     pi = np.random.rand(num_clusters)
#     pi /= np.sum(pi)

#     iteration = 0
#     while iteration < max_iterations:
#         old_lambdas = lambdas.copy()

#         # E-step
#         responsibilities = expectation_step(data, lambdas, pi)

#         # M-step
#         lambdas, pi = maximization_step(data, responsibilities)

#         # Check for convergence
#         if np.linalg.norm(lambdas - old_lambdas) < tolerance:
#             break

#         iteration += 1

#     return lambdas, pi, responsibilities

# # Example usage:
# # Assuming you have 'data' as a NumPy array with shape (num_samples, num_features)
# # where each row contains the counts of a data point, and 'num_clusters' is the desired number of clusters.
# # lambdas, pi, responsibilities = two_way_poisson_mixture_model(data, num_clusters)


import numpy as np

def probability_mass_function(d, lambd):
    return np.exp(-lambd) * np.power(lambd, d) / np.math.factorial(d)

def log_likelihood(data, weights, lambdas):
    n_components = len(weights)
    log_likelihoods = np.zeros(len(data))
    for i in range(n_components):
        log_likelihoods += weights[i] * probability_mass_function(data, lambdas[i])
    return np.sum(np.log(log_likelihoods))

def expectation_maximization(data, n_components, max_iter=100, tol=1e-6):
    n_samples = len(data)
    weights = np.ones(n_components) / n_components
    lambdas = np.random.rand(n_components)

    for _ in range(max_iter):
        # E-step
        responsibilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            responsibilities[:, i] = weights[i] * probability_mass_function(data, lambdas[i])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step
        new_lambdas = np.zeros(n_components)
        for i in range(n_components):
            new_lambdas[i] = np.sum(responsibilities[:, i] * data) / np.sum(responsibilities[:, i])

        # Check for convergence
        if np.all(np.abs(new_lambdas - lambdas) < tol):
            break

        lambdas = new_lambdas

    return weights, lambdas

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.poisson(lam=5, size=1000)  # Simulated Poisson data with mean 5

    n_components = 2
    weights, lambdas = expectation_maximization(data, n_components)

    print("Estimated Weights:", weights)
    print("Estimated Lambdas:", lambdas)
    print("Log Likelihood:", log_likelihood(data, weights, lambdas))
