# standard modules
from math import sqrt

# application specific modules
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt import solvers
from cvxopt.solvers import qp
import numpy as np

# custom modules
import params
import portfolio

def optimize(S=None):
    """
    
    Parameters
    ----------
    
    Returns
    -------
    
    
    """
    # turn off display of optimizations
    solvers.options['show_progress'] = False

    # get the portfolio parameters
    port_params = params.get_portfolio_params();
    bench_params = params.get_bench_params();

    # instantiate the porfolio object
    port = portfolio.Portfolio(port_params, bench_params)

    # get the required metrix for the opmization
    n = port.get_portfolio_size()
    
    if S is None:
        S = matrix(port.get_covariance_matrix().as_matrix())
    
    pbar = matrix(port.get_expected_stock_returns().as_matrix())

    # n x n matrix of zeros
    G = matrix(0.0, (n,n)) #original

    # diagonal matrix with -1.0 in the diagonal
    G[::n+1] = -1.0 #new

    # n x 1 matrix of 0.0s
    # this appears to be the constraint that x >= 0
    h = matrix(0.0, (n,1)) # original

    # 1 x n matrix of 1.0s
    # this appears to be the constraint that 1Tx = 1
    A = matrix(1.0, (1,n))

    # 1 x 1 matrix of 1.0s
    b = matrix(1.0)

    # Compute trade-off.
    N = 1

    mus = [ 10**(5.0*t/N-1.0) for t in xrange(N) ]

    # optimal portfolios for given mus
    portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]

    returns = [ dot(pbar,x) for x in portfolios ]
    risks = [ sqrt(dot(x, S*x)) for x in portfolios ]
    
    return portfolios