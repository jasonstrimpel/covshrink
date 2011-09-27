# standard modules
from math import sqrt

# application specific modules
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt import solvers
from cvxopt.solvers import qp
import pylab

# custom modules
import params
import portfolio

class Optimize(object):
    
    def __init__(self):        
        pass

# turn off display of optimizations
solvers.options['show_progress'] = False

# get the portfolio parameters
port_params = params.get_portfolio_params();
bench_params = params.get_bench_params();

# instantiate the porfolio object
port = portfolio.Portfolio(port_params, bench_params)

# get the required metrix for the opmization
n = port.get_portfolio_size()
S = matrix(port.get_covariance_matrix().as_matrix())
pbar = matrix(port.get_expected_stock_returns().as_matrix())
print pbar
# n x n matrix of zeros
G = matrix(0.0, (n,n))

# diagonal matrix with -1.0 in the diagonal
G[::n+1] = -1.0

# n x 1 matrix of 0.0s
# this appears to be the constraint that x >= 0
h = matrix(0.0, (n,1))

# 1 x n matrix of 1.0s
# this appears to be the constraint that 1Tx = 1
A = matrix(1.0, (1,n))

# 1 x 1 matrix of 1.0s
b = matrix(1.0)

# Compute trade-off.
N = 100

mus = [ 10**(5.0*t/N-1.0) for t in xrange(N) ]

# optimal portfolios for given mus
portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]

returns = [ dot(pbar,x) for x in portfolios ]
risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

# Plot trade-off curve and optimal allocations.
pylab.figure(1, facecolor='w')
pylab.plot(risks, returns)
pylab.xlabel('standard deviation')
pylab.ylabel('expected return')
#axis([xmin, xmax, ymin, ymax])
pylab.axis([min(risks)-0.0002, max(risks)+0.0002, min(returns)-0.0002, max(returns)+0.0002])
pylab.title('Risk-return trade-off curve (fig 4.12)')
#pylab.yticks([0.00, 0.05, 0.10, 0.15])

#pylab.show()

'''
pylab.figure(2, facecolor='w')
c1 = [ x[0] for x in portfolios ]
c2 = [ x[0] + x[1] for x in portfolios ]
c3 = [ x[0] + x[1] + x[2] for x in portfolios ]
c4 = [ x[0] + x[1] + x[2] + x[3] for x in portfolios ]
pylab.fill(risks + [.20], c1 + [0.0], '#F0F0F0')
pylab.fill(risks[-1::-1] + risks, c2[-1::-1] + c1, facecolor = '#D0D0D0')
pylab.fill(risks[-1::-1] + risks, c3[-1::-1] + c2, facecolor = '#F0F0F0')
pylab.fill(risks[-1::-1] + risks, c4[-1::-1] + c3, facecolor = '#D0D0D0')
pylab.axis([0.0, 0.2, 0.0, 1.0])
pylab.xlabel('standard deviation')
pylab.ylabel('allocation')
pylab.text(.15,.5,'x1')
pylab.text(.10,.7,'x2')
pylab.text(.05,.7,'x3')
pylab.text(.01,.7,'x4')
pylab.title('Optimal allocations (fig 4.12)')
pylab.show()
'''