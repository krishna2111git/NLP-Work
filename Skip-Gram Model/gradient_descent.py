import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Cost function computation for convergence check
def compute_cost(center_vec, context_vec, neg_sample_vecs):
    dot_pos = np.dot(center_vec, context_vec)
    cost_pos = -np.log(sigmoid(dot_pos))  # Cost for the positive pair

    cost_neg = 0
    for neg_vec in neg_sample_vecs:
        dot_neg = np.dot(center_vec, neg_vec)
        cost_neg += -np.log(sigmoid(-dot_neg))  # Cost for each negative sample

    return cost_pos + cost_neg





# Gradient computation and vector updates
def update_vectors(center_vec, context_vec, neg_sample_vecs, learning_rate=0.01):
    # Positive pair gradient
    dot_pos = np.dot(center_vec, context_vec)
    sigma_pos = sigmoid(dot_pos)
    grad_center_pos = (sigma_pos - 1) * context_vec
    grad_context = (sigma_pos - 1) * center_vec

    # Negative samples gradient
    grad_center_neg = np.zeros_like(center_vec)
    grad_neg_samples = []
    for neg_vec in neg_sample_vecs:
        dot_neg = np.dot(center_vec, neg_vec)
        sigma_neg = sigmoid(-dot_neg)
        grad_center_neg += (1 - sigma_neg) * neg_vec
        grad_neg_samples.append((1 - sigma_neg) * center_vec)

    # Total gradient for center vector
    grad_center = grad_center_pos + grad_center_neg

    # Update vectors
    center_vec -= learning_rate * grad_center
    context_vec -= learning_rate * grad_context
    for i, grad_neg in enumerate(grad_neg_samples):
        neg_sample_vecs[i] -= learning_rate * grad_neg

    return center_vec, context_vec, neg_sample_vecs




# Gradient descent with convergence check
def train_skip_gram(center_vec, context_vec, neg_sample_vecs, learning_rate=0.01, 
                    max_iterations=1000, tolerance=1e-6):
    iteration = 0
    prev_cost = float('inf')

    while iteration < max_iterations:
        # Compute the current cost
        cost = compute_cost(center_vec, context_vec, neg_sample_vecs)

        # Check convergence
        if abs(prev_cost - cost) < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        # Update vectors using gradient descent
        center_vec, context_vec, neg_sample_vecs = update_vectors(
            center_vec, context_vec, neg_sample_vecs, learning_rate
        )

        prev_cost = cost
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")

    return center_vec, context_vec, neg_sample_vecs


