import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_cost(X, y, theta):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    y_hat = sigmoid(X_b.dot(theta))
    J_theta = -1/m* (y.T.dot (np.log(y_hat))+(1-y).T.dot(np.log(1-y_hat)))
    return J_theta

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    J_history = np.zeros(num_iters)
    for i in range (num_iters):
        h_theta = sigmoid(X_b.dot(theta))
        J_dif = 1/m*X_b.T.dot(h_theta-y)
        theta = theta-alpha*J_dif
        J_history[i] = compute_logistic_cost(X,y,theta)
    return theta ,J_history

    

if __name__ == "__main__":
    X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ])

    # Logic NOT Test Case
    theta = np.array([0.3785, 0.4246, 0.4246], dtype=float)
    y_not = np.array([0,1,1,1])
    y_hat = sigmoid(0.3785)
    J = compute_logistic_cost(X, y_not, theta)
    # print(y_hat)
    print("Test Case 1: Zero Weight Logic NOT")
    print(f"{'Expected Cost:':<20} {0.6931:.4f}")
    print(f"{'Computed Cost:':<20} {J:.4f}\n")

    # # Test Gradient Descent with Logic NOT

    # # Test 1 Iteration
    # theta, J_history = gradient_descent(X, y_not, np.array([0.0, 0.0]), 0.1, 1)
    # print("Test Case 2: Logic NOT - 1 Iteration")
    # print(f"{'Expected Weights:':<20} [0.0000, -0.0250]")
    # print(f"{'Updated Weights:':<20} {[f'{w:.4f}' for w in theta]}")
    # print(f"{'Expected Cost:':<20} {0.6869:.4f}")
    # print(f"{'Computed Cost:':<20} {J_history[-1]:.4f}\n")

    # # Test 2 Iterations
    # theta, J_history = gradient_descent(X, y_not, np.array([0.0, 0.0]), 0.1, 2)
    # print("Test Case 3: Logic NOT - 2 Iterations")
    # print(f"{'Expected Weights:':<20} [0.0003, -0.0497]")
    # print(f"{'Updated Weights:':<20} {[f'{w:.4f}' for w in theta]}")
    # print(f"{'Expected Cost:':<20} {0.6809:.4f}")
    # print(f"{'Computed Cost:':<20} {J_history[-1]:.4f}\n")