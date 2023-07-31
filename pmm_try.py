import numpy as np

def poisson_pdf(x, lambd):
    return np.exp(-lambd) * np.power(lambd, x) / np.math.factorial(x)

def log_likelihood(data, weights, lambdas):
    n_components = len(weights)
    log_likelihoods = np.zeros(len(data))
    for i in range(n_components):
        log_likelihoods += weights[i] * poisson_pdf(data, lambdas[i])
    return np.sum(np.log(log_likelihoods))

def expectation_maximization(data, n_components, max_iter=100, tol=1e-6):
    n_samples = len(data)
    weights = np.ones(n_components) / n_components
    lambdas = np.random.rand(n_components)

    for _ in range(max_iter):
        # E-step
        responsibilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            responsibilities[:, i] = weights[i] * poisson_pdf(data, lambdas[i])

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
