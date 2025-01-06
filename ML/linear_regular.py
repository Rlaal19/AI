import numpy as np
# 1
def compute_cost_regularized(X, y, theta, lambda_):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    y_hat = X_b.dot(theta)
    R = np.identity(len(theta))
    R[0] = 0
    R_theta = R.dot(theta)
    J_theta =1/(2*m)*((y_hat-y).T.dot(y_hat-y))+lambda_/(2*m)*theta.T.dot(R_theta)
    print(J_theta)
    return J_theta
# 2
def gradient_descent_linear(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        Regular = lambda_/m*theta
        Regular[0] = 0 
        theta = theta-alpha*(1/m*X_b.T.dot(X_b.dot(theta)-y)+Regular)
        J_history[i] = compute_cost_regularized(X, y, theta, lambda_)
    return theta,J_history