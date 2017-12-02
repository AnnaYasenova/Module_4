import pandas as pd
import numpy as np

def mserror(y, y_pred):
    return np.sum(((y - y_pred) ** 2) / y.shape[0])

# calculates vector w
def normal_equation(X, y):
    return np.dot(np.linalg.pinv(X), y)

# returns linear prediction
def linear_prediction(X, w):
    return np.dot(X, w)

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    # gradient array for every param
    grad = np.zeros((X.shape[1], 1))

    xRowToCol = X[train_ind].reshape(-1, 1)

    # calculating gradient for every param
    for i in range(X.shape[1]):
        grad[i] = xRowToCol[i] * (np.sum(xRowToCol * w) - y[train_ind])

    return (2 * eta / X.shape[0]) * grad


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # distance between weight vector is a big number
    weight_dist = np.inf
    # weight vector
    w = w_init
    # array with errors on every iteration
    errors = []
    # iterations counter
    iter_num = 0

    np.random.seed(seed)

    while weight_dist > min_weight_dist and iter_num < max_iter:

        random_ind = np.random.randint(X.shape[0])

        last = w
        w = last - stochastic_gradient_step(X, y, w, random_ind, eta=eta)
        weight_dist = np.linalg.norm(last - w)
        error = mserror(y, linear_prediction(X, w))
        errors.append(error)
        iter_num += 1

    return w, errors

def main():
    adver_data = pd.read_csv('advertising.csv')

    print(adver_data.head(5))
    print(adver_data.describe())

    X = adver_data[['TV', 'Radio', 'Newspaper']].values
    y = adver_data[['Sales']].values
    print('Array X. Shape: ', X.shape, '\n', X)
    print('Array y. Shape: ', y.shape, '\n', y)

    means, stds = X.mean(axis = 0), X.std(axis = 0)
    print('Means: ', means)
    print('Stds: ', stds)

    # normalize
    X = (X - means) / stds
    print('Array X. Shape: ', X.shape, '\n', X)

    # once column
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    print('Array X. Shape: ', X.shape, '\n', X)

    norm_eq_weights = normal_equation(X, y)
    print('Vector of weights: \n', norm_eq_weights)

    stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros((X.shape[1], 1)), max_iter = 1e5)

    print('The vector of weights to which the method converges\n',stoch_grad_desc_weights)
    print('Mean square error of forecasting Sales values ​​in the form of a linear model with weights found using gradient descent')
    print(mserror(y, linear_prediction(X, stoch_grad_desc_weights)))

if __name__ == '__main__':
    main()