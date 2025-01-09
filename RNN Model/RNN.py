import numpy as np
import matplotlib.pyplot as plt
import keras.api.models as mod
import keras.api.layers as lay

pitch = 20
step = 1
N = 100
n_train = int(N*0.9)

def gen_data(x):
    return (x%pitch)/pitch

def convertToMatrix(data, step=1):
    X,Y = [],[]
    for i in range(len(data)-step):
        d = i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

# y คือ ค่า y, u คือจำนวนโหนดใน layer, e คือจำนวนการเทรน
def model(y,u,e):
    train,test =y[0:n_train],y[n_train:N]
    x_train, y_train = convertToMatrix(train,step)
    x_test, y_test = convertToMatrix(test,step)

    print("Dimension before: " ,train.shape,test.shape)
    print("Dimension after: " ,x_train.shape,x_test.shape)


    model = mod.Sequential()
    model.add(lay.SimpleRNN(units=u, input_shape = (step,1), activation = "tanh"))
    model.add(lay.Dense(units=1))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=e, batch_size=1, verbose=1)
    predict1 = model.predict(x_train)
    predict2 = model.predict(x_test)
    predict = np.append(predict1, predict2)
    return predict, x_train

if __name__ == "__main__":
    t = np.arange(1, N+1)
    y1 = [gen_data(i) for i in t]
    y1 = np.array(y1)

    y2 = np.sin(0.05*t*10) +0.8 * np.random.rand(N)
    y2 = np.array(y2)
    # graph 1
    predict1, x_train1 = model(y1,300,300)
    plt.figure(figsize=(10, 6))
    plt.plot(y1, label='Original')
    plt.axvline(x=len(x_train1), color='r')
    plt.plot(predict1, label='Predicted')    
    plt.xlabel('Timestamp')
    plt.ylabel('Electricity Consumption')
    plt.title('Electricity Consumption Prediction using RNN (PyTorch)')
    plt.legend()
    plt.show()

    # graph 2
    predict2, x_train2 = model(y2,300,300)
    plt.figure(figsize=(10, 6))
    plt.plot(y2, label='Original')
    plt.axvline(x=len(x_train2), color='r')
    plt.plot(predict2, label='Predicted')    
    plt.xlabel('Timestamp')
    plt.ylabel('Electricity Consumption')
    plt.title('Electricity Consumption Prediction using RNN (PyTorch)')
    plt.legend()
    plt.show()

    



