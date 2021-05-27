from sklearn.utils import shuffle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def svm_with_sgd(X, y, lam=0.0, epochs=1000, l_rate=0.01, sgd_type='practical'):
    np.random.seed(2)
    m, d = X.shape[0], X.shape[1]
    w = np.random.uniform(0, 1, d)
    b = np.random.uniform(0, 1, 1)
    sbg_w = lambda w, x, y: -y * x + 2 * lam * w
    if sgd_type == 'practical':
        for i in range(epochs):
            perm = np.random.permutation(m)
            for observation in perm:
                xi, yi = X[observation], y[observation]
                if 1 - yi*b - yi * (w @ xi.T) > 0:
                    w = w - l_rate * sbg_w(w, xi, yi)
                    b = b - l_rate * (-yi)
                else:
                    w = w - l_rate * 2 * lam * w
        return w, b
    else:
        wt =[w]
        bt = [*b]
        for i in range(m * epochs):
            index = int(np.random.choice(m, 1))
            xi, yi = X[index], y[index]
            if 1 - bt[i]*yi - yi * (wt[i] @ xi.T) > 0:
                wt.append(wt[i] - l_rate * sbg_w(wt[i], xi, yi))
                bt.append(bt[i] - l_rate * (-yi))
            else:
                wt.append(wt[i] - l_rate * 2 * lam * wt[i])
                bt.append(bt[i])
        wtm, btm = np.mean(np.array(wt), axis=0), np.mean(bt)
        return wtm, btm


def calculate_error(w, b, X, y):
    y_pred = []
    for x in X:
        y_pred.append(np.sign(w @ x.T + b))
    accuracy = accuracy_score(y, y_pred)
    return 1-accuracy


def main():
    X, y = load_iris(return_X_y=True)
    X = X[y != 0]
    y = y[y != 0]
    y[y==2]= -1
    X= X[:, 2:4]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    lam = [0, 0.05, 0.1, 0.2, 0.5]
    error_train = []
    error_test = []
    margins = []
    for l in lam:
        w, b = svm_with_sgd(X_train, y_train, lam=l)
        error_train.append(calculate_error(w, b, X_train, y_train))
        error_test.append(calculate_error(w, b, X_val, y_val))
        margins.append(1/np.linalg.norm(w))

    #plot error
    fig, ax = plt.subplots()
    x = np.arange(len(lam))
    width = 0.3
    rects1= ax.bar(x - width/2, error_train , width, color = 'b', label = 'train error')
    rects2 =ax.bar(x + width/2, error_test, width, color='r', label='test error')
    ax.set_xticks(x)
    ax.set_xticklabels(lam)
    ax.set_ylabel('Errors')
    ax.set_title("error train vs error test by lam")
    ax.legend()
    fig.tight_layout()
    plt.show()

    #plot margin
    fig, ax = plt.subplots()
    x = np.arange(len(lam))
    width = 0.3
    ax.bar(x, margins, width, color = 'g', label = 'margin width')
    ax.set_xticks(x)
    ax.set_xticklabels(lam)
    ax.set_ylabel('width of margin')
    ax.set_title("margin by lam")
    ax.legend()
    fig.tight_layout()
    plt.show()
