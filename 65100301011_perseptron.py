# นางสาวปริชญา  นาสำแดง 6510301011
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
def dataSample():
    X1,y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0,2.0),
                    cluster_std=0.25,
                    random_state=69)
    
    X2,y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0,3.0),
                    cluster_std=0.25,
                    random_state=69)
    return X1,y1,X2,y2

def check_class(x): #จำแนก class
    return 1 if x >= 0 else 0
def perseptron(X, y, learning_rate=0.1, epochs=100):
    weights = np.random.rand(3)
    print(weights)
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            prediction = check_class(np.dot(X[i], weights))  # คำนวณผลลัพธ์
            error = y[i] - prediction  # คำนวณ error
            weights += learning_rate * error * X[i]  # ปรับปรุงน้ำหนัก
    return weights


def decision_func(x1,x2):
    return x1 + x2 -0.



def plot_dataset(): #กราฟแสดงตัวอย่างข้อมูล
    fig = plt.figure()
    fig.suptitle("Data Sample")
    plt.scatter(X1[:,0], X1[:,1], c='red',linewidths=1,alpha=0.6,label="Class 1")
    plt.scatter(X2[:,0], X2[:,1], c='blue',linewidths=1,alpha=0.6,label="Class 2")
    plt.xlabel('Feature 1', fontsize=10)
    plt.ylabel('Feature 2', fontsize=10)
    plt.grid(True, axis='both')
    plt.legend(loc='lower right')
    plt.show()

def plot_decision():
    x1_range = np.linspace(-1, 2, 500)
    x2_range = np.linspace(-1, 2, 500)
    x1_grid,x2_grid = np.meshgrid(x1_range,x2_range)
    g_value = decision_func(x1_grid,x2_grid)
    plt.figure()
    plt.contourf(x1_grid, x2_grid, g_value, levels=[-np.inf, 0, np.inf], colors= ['red','blue'], alpha=0.5)
    plt.contour(x1_grid, x2_grid, g_value, levels=[0], colors='black', linewidths=2)
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title("Decision Plane")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, y, weights): #กราฟแสดงเส้นตัวแบ่ง class
    fig = plt.figure()
    fig.suptitle("Perceptron")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='purple',alpha=0.6, label='Class 1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='yellow',alpha=0.6, label='Class 2')
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 120)
    y_vals = -(weights[0] * x_vals + weights[2]) / weights[1]  # Decision Boundary
    plt.plot(x_vals, y_vals, color='black', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='lower right')
    plt.grid(True, axis='both')
    plt.show()


if __name__ == "__main__":
    X1,y1,X2,y2 = dataSample()
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(y1.shape[0]), np.ones(y2.shape[0])))  # Class 1(y1) = 0, Class 2(y2) = 1
    X = np.c_[X, np.ones(X.shape[0])]# add bias term in x
    weights = perseptron(X, y)
    plot_dataset()
    plot_decision()
    plot_decision_boundary(X, y, weights)



