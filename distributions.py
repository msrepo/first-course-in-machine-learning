import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.special import gamma
from math import factorial

# discrete distributions


def bernoulli(q, x):
    """
    q: probability that the binary random variable takes value 1
    x: value of the binary random variable X, either 0 or 1
    """
    return q * x + (1 - q) * (1 - x)


def binomial(q, y, N):
    """
    q : probability that the binary random variable takes value 1
    N : no of events
    y : no of successes
    """
    return comb(N, y) * q**y * (1 - q)**(N - y)


def multinomial(q, y):
    """
    q : probability vector that the N-ary random variable takes value i 1<=i<=N
    y : vector that counts successes that random variable takes value i
    N : sum of counts of successes that random variable takes value i \sum{y}

    >>> multinomial([0.166,0.166,0.166,0.166,0.166,0.166],[1,2,3,4,5,6])
    0.00009
    """
    N = np.sum(y)
    p_y = 1.0
    product_yj = 1.0
    for q_, y_ in zip(q, y):
        p_y = p_y * (q_ ** y_)
        product_yj = product_yj * factorial(y_)
    return (factorial(N) / product_yj) * p_y

# continuous density functions


def beta_distribution(alpha, beta, bias):
    return gamma(alpha + beta) / (gamma(alpha) * gamma(beta)) * bias**(alpha - 1) * (1 - bias)**(beta - 1)


# plot beta distribution

r = np.linspace(0, 1, 100)
fig = plt.figure()
alpha, beta = 3, 7
plt.plot(r, beta_distribution(alpha, beta, r),
         label=r'$\alpha$={},$\beta$={}'.format(alpha, beta))
alpha, beta = 1, 1
plt.plot(r, beta_distribution(alpha, beta, r),
         label=r'$\alpha$={},$\beta$={}'.format(alpha, beta))
alpha, beta = 70, 30
plt.plot(r, beta_distribution(alpha, beta, r),
         label=r'$\alpha$={},$\beta$={}'.format(alpha, beta))
plt.legend()
plt.xlabel('r')
plt.ylabel('p(r)')
plt.title('Beta distribution')
plt.show()

# plot binomial distributions
N = 50
q = 0.7
p_y = [binomial(q, y, N) for y in range(0, 50 + 1)]
y = list(range(0, 50 + 1))
fig = plt.figure()
plt.bar(y, p_y)
plt.xlabel('y:no of successes')
plt.ylabel('p(y) : probability of y successes in 50 trials')
plt.title('Binomial Distribution (q = 0.7, N = 50 )')
plt.show()
np.testing.assert_almost_equal(multinomial(
    [0.16666666667, 0.16666666667, 0.16666666667, 0.16666666667, 0.16666666667, 0.16666666667], [1, 2, 3, 4, 5, 6]), 0.00009, decimal=5)
