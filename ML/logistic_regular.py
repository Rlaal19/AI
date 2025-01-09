import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


# 3
def compute_logistic_cost_regularized(X, y, theta, lambda_):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    y_hat = sigmoid(X_b.dot(theta))
    R = np.identity(theta.shape[0])
    R[0] = 0
    logis = -1 / m * (y.T.dot(np.log(y_hat)) + (1 - y).T.dot(np.log(1 - y_hat)))
    R_theta = R.dot(theta)
    Regular = lambda_ / (2 * m) * theta.T.dot(R_theta)
    cost = logis + Regular
    return cost


# 4


def gradient_descent_logistic(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        Regular = (lambda_ / m) * theta
        Regular[0] = 0
        y_hat = sigmoid(X_b.dot(theta))
        J_dif = 1 / m * X_b.T.dot(y_hat - y) + Regular
        theta = theta - alpha * J_dif
        J_history[i] = compute_logistic_cost_regularized(X, y, theta, lambda_)
    return theta, J_history


if __name__ == "__main__":
    # X = np.array([
    #     [1],
    #     [2],
    #     [3]
    # ])

    # theta = np.array([0.5733, 1.319983])
    # y = np.array([2, 4, 6])

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    theta = np.array([0.0, 0.0, 0.0])
    y = np.array([0, 1, 1, 1])

    theta, J = gradient_descent_logistic(X, y, theta, 0.1, 1, 1.0)
    print("Test Case 1: Logic OR - 1 Iteration with Regularization")
    print(f"{'Expected Weights:':<30} [0.0250, 0.0250, 0.0250]")
    print(f"{'Updated Weights:':<30} {[f'{w:.4f}' for w in theta]}")
    print(f"{'Expected Cost:':<30} {0.6749:.4f}")
    print(f"{'Computed Cost:':<30} {J[-1]:.4f}\n")

    theta, J = gradient_descent_logistic(X, y, theta, 0.1, 1, 1.0)
    print("Test Case 2: Logic OR - 2 Iterations with Regularization")
    print(f"{'Expected Weights:':<30} [0.0488, 0.0486, 0.0486]")
    print(f"{'Updated Weights:':<30} {[f'{w:.4f}' for w in theta]}")
    print(f"{'Expected Cost:':<30} {0.6586:.4f}")
    print(f"{'Computed Cost:':<30} {J[-1]:.4f}\n")
