from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt import solvers
from cvxopt.solvers import qp
import pylab

from scipy import linalg

solvers.options['show_progress'] = False

# Problem data.

# number of assets
n = 10

'''
# covar or cor matrix
S = matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ],
            [ 6e-3,  1e-2,  0.0,     0.0 ],
            [-4e-3,  0.0,   2.5e-3,  0.0 ],
            [ 0.0,   0.0,   0.0,     0.0 ]])

# expected returns
pbar = matrix([.12, .10, .07, .03])
'''
# covariance matrix from portfolio.py
S = matrix([[ 0.00154474, 0.00079555, 0.00099691, 0.00052596, 0.0005363,  0.00062005,  0.00064031, 0.00037494, 0.00018826, 0.00132809],
 [ 0.00079555, 0.00287429, 0.00058536, 0.00091774, 0.00046885, 0.00110434,  0.00137141, 0.00046724, 0.00030414, 0.0016615 ],
 [ 0.00099691, 0.00058536, 0.00155757, 0.00056336, 0.00052395, 0.00060104,  0.00057223, 0.00021365, 0.00017057, 0.00130247],
 [ 0.00052596, 0.00091774, 0.00056336, 0.00126312, 0.00031941, 0.00088137,  0.00024493, 0.00025136, 0.00011519, 0.00135475],
 [ 0.0005363,  0.00046885, 0.00052395, 0.00031941, 0.00054093, 0.00045649,  0.00042927, 0.00021928, 0.00016835, 0.00093471],
 [ 0.00062005, 0.00110434, 0.00060104, 0.00088137, 0.00045649, 0.00133081,  0.00060353, 0.0003967,  0.00024983, 0.00168281],
 [ 0.00064031, 0.00137141, 0.00057223, 0.00024493, 0.00042927, 0.00060353,  0.00468731, 0.00059557, 0.00020384, 0.00078669],
 [ 0.00037494, 0.00046724, 0.00021365, 0.00025136, 0.00021928, 0.0003967,  0.00059557, 0.00082333, 0.00017191, 0.00066816],
 [ 0.00018826, 0.00030414, 0.00017057, 0.00011519, 0.00016835, 0.00024983,  0.00020384, 0.00017191, 0.00036348, 0.0004505 ],
 [ 0.00132809, 0.0016615,  0.00130247, 0.00135475, 0.00093471, 0.00168281,  0.00078669, 0.00066816, 0.0004505,  0.00530036]])

# expected returns from portfolio.py
pbar = matrix([ 0.09, 0.05, 0.15, 0.08, 0.15, 0.15, 0.06, 0.08, 0.1, 0.05])

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

for i in portfolios:
    print i

returns = [ dot(pbar,x) for x in portfolios ]
risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

# Plot trade-off curve and optimal allocations.
pylab.figure(1, facecolor='w')
pylab.plot(risks, returns)
pylab.xlabel('standard deviation')
pylab.ylabel('expected return')
#axis([xmin, xmax, ymin, ymax])
pylab.axis([min(risks), max(risks), min(returns), max(returns)])
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