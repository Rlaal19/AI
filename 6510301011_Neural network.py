# นางสาวปริชญา  นาสำแดง 6510301011
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import layers

def dataSample():
    X1,y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0,2.0),
                    cluster_std=0.75,
                    random_state=69)
    
    X2,y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0,3.0),
                    cluster_std=0.75,
                    random_state=69) 
    return X1,X2,y1,y2

def nomalize_data(x):
    ss = StandardScaler()
    X_ss = ss.fit_transform(x)
    return X_ss


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

# Neural network
def Neural(X,y):
    #nomalize data
    X = nomalize_data(X)

    #split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    input_shape = X_train.shape[1]

    #make model
    model = Sequential([
    layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'), #กำหนดให้ใช้ 32 node(เลขฐาน 2)
        layers.Dense(1, activation='sigmoid') #กำหนด 1 node เพราะแบ่ง y เป็น 2 class คือ 0 กับ 1
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #train model data
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'loss: {score[0]}')
    print(f'accuracy: {score[1]}')

    prediction = model.predict(X_test)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(y_pred[:5])

    #plot data
    plot_decision_boundary(model,X_train,y_train)
    

def plot_decision_boundary(model, X, Y):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(y_min, y_max, h), np.arange(x_min, x_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)

    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_points, edgecolors='k')

    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()



if __name__ == "__main__":
    X1,X2,y1,y2 = dataSample()
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(y1.shape[0]), np.ones(y2.shape[0])))  # Class 1(y1) = 0, Class 2(y2) = 1
    print('x = ',X)
    print('y = ',y)
    plot_dataset()
    Neural(X,y)
