import numpy as np
import os
import struct
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """
    Load MNIST data from path
    :param path: str
    :param kind: str, 'train' for training set, 't10k' for test set (validation set)
    :return images: numpy array, rows are reshaped pics
    :return labels: numpy array, labels
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784) # reshape into row vector for each pic
    return images, labels.reshape(labels.shape[0], 1)


def plot_set(x_train, y_train):
    """
    Plot the data set
    :param x_train: numpy array, rows are reshaped pics
    :param y_train: numpy array, labels
    :return None
    """
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        for j in range(x_train.shape[0]):
            if y_train[i][j] == 1:
                img = x_train[0:784, j].reshape(28, 28)
                ax[i].imshow(img, cmap='Greys', interpolation='nearest')
                break

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def sigmoid(z):
    """
    sigmoid function
    :param z: numpy array, vector
    :return a: numpy array, vector
    """
    a = np.divide(1, (1+np.exp(z)))
    return a


def dsigmoid(z):
    """
    derivative of sigmoid function
    :param z: numpy array, vector
    :return dg: numpy array, vector
    """
    dg = np.divide(np.exp((-1)*z), (1+np.exp((-1)*z))**2)
    return dg


def relu(z):
    """
    ReLU function
    :param z: numpy array, vector
    :return: a: numpy array, vector
    """
    tmpa = z > 0
    a = np.multiply(tmpa, z)
    return a


def drelu(z):
    """
    derivative of ReLU function
    :param z: numpy array, vector
    :return: dg: numpy array, vector
    """
    dg = z > 0
    return dg


def leaky_relu(leaky, z):
    """
    Leaky ReLU function
    :param z: numpy array, vector
    :param leaky: float, usually very small, the derivative when z<0
    :return: a: numpy array, vector
    """
    tmpa = z <= 0
    tmpa = tmpa * leaky
    tmpb = z > 0
    tmpc = tmpa + tmpb
    a = np.multiply(tmpc, z)
    return a


def dleaky_relu(leaky, z):
    """
    derivative of Leaky ReLU function
    :param leaky: float, leaky
    :param z: numpy array, vector
    :return: dg: numpy array, vector
    """
    tmpa = z <= 0
    tmpa = tmpa * leaky
    tmpb = z > 0
    dg = tmpa + tmpb
    return dg


def tanh(z):
    '''
    tanh function
    :param z: numpy array, vector
    :return: a: numpy array, vector
    '''
    a = np.divide(np.exp(z)-np.exp((-1)*z), np.exp(z)+np.exp((-1)*z))
    return a


def dtanh(z):
    """
    derivative of tanh funtion
    :param z: numpy array, vector
    :return: dg: numpy array, vector
    """

    dg = (1/(np.exp(z) + np.exp((-1)*z)))**2
    dg = 2 * dg
    return dg


def layer_forward(w_l, b_l, a_l_minus_1, activation=sigmoid, leaky=0.01):
    """
    layer for forward propagation
    :param w_l: numpy array, size n^[l]*n^[l-1]
    :param b_l: numpy array, size n^[l]*1
    :param a_l_minus_1: numpy array, size n^[l-1]*m
    :param activation: function, activation function
    :param leaky: float, value of leaky in activation function of Leaky ReLU
    :return: a_l: numpy array, size n^[l]*m
    :return: z_l: numpy array, size n^[l]*m (for storage)
    """
    z_l = np.dot(w_l, a_l_minus_1) + b_l
    if activation == leaky_relu:
        a_l = activation(leaky, z_l)
    else:
        a_l = activation(z_l)
    return a_l, z_l


def layer_backward(w_l, b_l, da_l, z_l, a_l_minus_1, activation=sigmoid, leaky=0.01):
    """
    layer for backward propagation
    :param w_l: numpy array, size n^[l]*n^[l-1]
    :param b_l: numpy array, size n^[l]*1
    :param da_l: numpy array, size n^[l]*m
    :param z_l: numpy array, size n^[l]*m
    :param activation: function, activation function
    :param leaky: float, value of leaky in activation function of Leaky ReLU
    :param a_l_minus_1: numpy array, size n^[l-1]*m
    :return:
    """
    if activation == sigmoid:
        dg = dsigmoid
    elif activation == relu:
        dg = drelu
    elif activation == leaky_relu:
        dg = dleaky_relu
    elif activation == tanh:
        dg = dtanh
    else:
        assert False
    dz_l = np.multiply(da_l, dg(z_l))
    dw_l = np.dot(dz_l, a_l_minus_1.T)
    # dw_l = np.dot(da_l, a_l_minus_1.T)
    db_l = (1/da_l.shape[1]) * np.sum(dz_l, axis=1, keepdims=True)
    da_l_minus_1 = np.dot(w_l.T, dz_l)
    return dw_l, db_l, da_l_minus_1


def dcost(y_train, y_predict):
    """
    derivative of cost function
    :param y_train: numpy array, label, size 10*m
    :param y_predict: numpy array, predict, size 10*m
    :return: da_L: numpy array, size 10*m
    """
    da_L = ((-1)*np.divide(y_train, y_predict) + np.divide(1-y_train, 1-y_predict)) / y_train.shape[1]
    return da_L


def cost(y_train, y_predict):
    """
    cost funtion
    :param y_train: numpy array, size 10*m
    :param y_predict: numpy array, size 10*m
    :return: cost: float, cost
    """
    cost = ((-1)*(y_train*np.log(y_predict)+(1-y_train)*np.log(1-y_predict))/y_train.shape[1])
    cost = np.sum(cost)
    return cost


def train_nn(x_train, y_train, x_test, y_test, iteration=10000, alpha=0.1):
    """
    train neural network
    :param x_train: numpy array, size n^[0]*m, each column is a sample
    :param y_train: numpy array, size 10*m, the index of 1s in each column is the label
    :param x_test: numpy array, size n^[0]*m_test, each column is a test sample
    :param y_test: numpy array, size 10*m_test, the index of 1s in each column is the label
    :param iteration: int, iteration of training
    :return: w_: list, consist of w_n
    :return: b_: list, consist of b_n
    """
    neuron = [28*28, 256, 128, 10]
    activation_func = [1, relu, relu, sigmoid]
    w_ = [1]
    b_ = [1]
    dw_ = [1]
    db_ = [1]
    a_ = [1]
    z_ = [1]
    da_ = [1]
    a_test = [1]
    cost_last = 0
    for i in range(1, len(neuron)):
        # random initialize
        w_.append(np.random.randn(neuron[i], neuron[i - 1])*np.sqrt(2/neuron[i-1]))
        b_.append(np.random.rand(neuron[i], 1))
        dw_.append(1)
        db_.append(1)
        a_.append(1)
        z_.append(1)
        da_.append(1)
        a_test.append(1)

    a_[0] = x_train
    a_test[0] = x_test
    for i in range(iteration):
        # forward propagation
        for j in range(1, len(neuron)):
            a_[j], z_[j] = layer_forward(w_[j], b_[j], a_[j-1], activation=activation_func[j])
        # compute da_L
        da_[len(neuron)-1] = dcost(y_train, a_[len(neuron)-1])
        # backward propagation
        for j in range(len(neuron)-1, 0, -1):
            dw_[j], db_[j], da_[j-1] = layer_backward(w_[j], b_[j], da_[j], z_[j], a_[j-1], activation=activation_func[j])
        # update w and b
        for j in range(1, len(neuron)):
            w_[j] = w_[j] + alpha * dw_[j]  # I have no idea why it's + here, but it's the only way it works...
            b_[j] = b_[j] + alpha * db_[j]  # same as the last line
        # compute cost
        cost_now = cost(y_train, a_[len(neuron)-1])
        # compute accuracy
        for j in range(1, len(neuron)):
            a_test[j], tmpz = layer_forward(w_[j], b_[j], a_test[j-1], activation=activation_func[j])
        predict_num = np.argmax(a_test[len(neuron)-1], axis=0)
        label_num = np.argmax(y_test, axis=0)
        match_num = (predict_num == label_num)
        acc = np.sum(match_num)/np.sum(y_test)
        print("training(iteration):" + str(round((i + 1) / iteration * 100, 2)) + "%", end=', ')
        print("cost:" + str(round(cost_now, 8)), end=', ')
        print("accuracy(test set):" + str(round(acc*100, 8)) + "%", end='\n')
        if abs(cost_last-cost_now) <= 1e-6:
            break
        else:
            cost_last = cost_now
    print("Training Complete!")
    return w_, b_


def label2ind(label):
    """
    transform label with number to matrix that index in each column is the label of the sample
    :param label: numpy array, 1*m
    :return: lb: numpy array, 10*m
    """
    lb = np.zeros([10, label.shape[1]])
    for i in range(label.shape[1]):
        lb[label[0][i]][i] = 1
    return lb


if __name__ == "__main__":
    path = "C:\\Users\\liuyu\\Desktop\\minist"
    img_train, label_train = load_mnist(path, 'train')
    img_test, label_test = load_mnist(path, 't10k')
    plot_set(img_train, label_train)
    label_train = label2ind(label_train.T)
    # normalize training set
    img_train = img_train / 255.0
    label_test = label2ind(label_test.T)
    # normalize test set
    img_test = img_test / 255.0
    W, b = train_nn(img_train.T, label_train, img_test.T, label_test, iteration=500, alpha=0.01)
    np.save("C:\\Users\\liuyu\\Desktop\\w_para", W)
    np.save("C:\\Users\\liuyu\\Desktop\\b_para", b)

