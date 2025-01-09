# import numpy as np
# # 1
# def compute_cost_regularized(X, y, theta, lambda_):
#     m = len(y)
#     X_b = np.c_[np.ones((X.shape[0],1)),X]
#     y_hat = X_b.dot(theta)
#     R = np.identity(len(theta))
#     R[0] = 0
#     R_theta = R.dot(theta)
#     J_theta =1/(2*m)*((y_hat-y).T.dot(y_hat-y))+lambda_/(2*m)*theta.T.dot(R_theta)
#     print(J_theta)
#     return J_theta
# # 2
# def gradient_descent_linear(X, y, theta, alpha, num_iters, lambda_):
#     m = len(y)
#     X_b = np.c_[np.ones((X.shape[0],1)),X]
#     J_history = np.zeros(num_iters)
#     for i in range(num_iters):
#         Regular = lambda_/m*theta
#         Regular[0] = 0 
#         theta = theta-alpha*(1/m*X_b.T.dot(X_b.dot(theta)-y)+Regular)
#         J_history[i] = compute_cost_regularized(X, y, theta, lambda_)
#     return theta,J_history

import numpy as np

def compute_cost(X,y,theta,lambda_):
    m = X.shape[0]
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    
    R = np.identity(theta.shape[0])
    R[0,0] = 0
    R_theta = R.dot(theta)
    
    y_hat = X_b.dot(theta)
    
    regular = lambda_/(2*m)*theta.T.dot(R_theta)
    linear = 1/(2*m) *((y_hat-y).T.dot(y_hat-y))
    
    cost = linear+regular
    return cost
def gradient_descent_linear(X, y, theta, alpha, num_iters, lambda_):
    m = X.shape[0]
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    J_his = np.zeros(num_iters)
    
    for i in range(num_iters):
        regular = lambda_/m*theta
        regular[0] = 0
        y_hat = X_b.dot(theta)
        theta = theta - alpha*(1/m*X_b.T.dot(y_hat-y)+regular)
        J_his[i] = compute_cost(X,y,theta,lambda_)
        
    return theta,J_his

if __name__ == "__main__":
    X = np.array([[0], [2]])
    y = np.array([0, 2])
    theta = np.array([0.0, 0.0])
    alpha = 0.1
    lambda_ = 1.0

    print("Test Case 1: Custom Data X = [0, 2], y = [0, 2]")
    for i in range(1, 4):
        theta, J = gradient_descent_linear(X, y, theta, alpha, 1, lambda_)
        print(f"Iteration {i}:")
        print(f"{'Updated Weights:':<25} {[f'{w:.4f}' for w in theta]}")
        print(f"{'Computed Cost:':<25} {J[-1]:.4f}\n")

# Test Case 1: Custom Data X = [0, 2], y = [0, 2]
# Iteration 1:
# Updated Weights:          ['0.1000', '0.2000']
# Computed Cost:            0.5750

# Iteration 2:
# Updated Weights:          ['0.1700', '0.3400']
# Computed Cost:            0.3668

# Iteration 3:
# Updated Weights:          ['0.2190', '0.4380']
# Computed Cost:            0.2647