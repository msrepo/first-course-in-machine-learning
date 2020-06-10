import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def read_olympics_data():
    filename = './data/olympics.mat'
    mat = scipy.io.loadmat(filename)
    print(mat['male100'].shape)
    print(mat['male100'])

    return mat['male100']


def plot_data(X, t, Xnew, tnew):
    plt.scatter(X, t, label='Original')
    plt.plot(Xnew, tnew, label='predicted')
    plt.title('Linear fitting')
    plt.legend()
    plt.show()


def evaluate_model_parameters(X, t, lambda_hyperparam=0.0):
    N = t.shape[0]
    w = np.linalg.inv(X.T @ X + N * lambda_hyperparam * np.eye(2)) @ X.T @ t
    print(w)
    return w


data = read_olympics_data()
X = np.array(data[:, 0], dtype='float').reshape(-1, 1)
X = np.hstack((np.ones_like(X), X))
t = np.array(data[:, 1], dtype='float').reshape(-1, 1)
print(X.shape, t.shape)
w = evaluate_model_parameters(X, t)
Xnew = np.array(np.linspace(X[0, 1], X[-1, 1])).reshape(-1, 1)  # columnize
Xnew = np.hstack((np.ones_like(Xnew), Xnew))
tnew = np.dot(Xnew, w)
plot_data(X[:, 1], t, Xnew[:, 1], tnew)



