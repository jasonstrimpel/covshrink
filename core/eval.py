# standard modules
from datetime import datetime
from datetime import date
from dateutil import relativedelta
import math
import time

# custom modules
import params2
import portfolio
import optimize as op
import numpy as np
import pandas
from cvxopt import matrix

def eval(type, index=30):
    """
    
    Parameters
    ----------
    
    Returns
    -------
    
    
    """
    # get the portfolio parameters
    port_params = params2.get_portfolio_params(index=index)

    # instantiate the porfolio object
    port = portfolio.Portfolio(port_params, proxy={"http": "http://proxy.jpmchase.net:8443"})

    # setup the periodicity
    roll = 60
    rollperiod = relativedelta.relativedelta(months=roll)
    outsample = 60
    outsampleperiod = relativedelta.relativedelta(months=outsample)

    # setup 
    dates = port.get_trading_dates()
    start = dates[0] + rollperiod
    end = dates[-1]

    delta = relativedelta.relativedelta(end, start)
    periods = (delta.years * 12) + delta.months

    portvalue = port.get_portfolio_historic_position_values()
    
    # constant benchmark weights
    #returns = port.get_portfolio_historic_returns()
    active = port.get_active_returns()

    expected_excess_returns = port.get_expected_excess_stock_returns()

    e = []

    for i in xrange(roll, periods+roll+1):
        
        # setup the dates to calculate returns for the covariance matrixes
        start = dates[i-roll]
        end = dates[i]

        active_returns = active.ix[start:end]
        
        # compute the sample covariance matrix, cov of active returns
        cov = port.get_covariance_matrix(active_returns)
        
        # actual realized returns
        y = ((portvalue.ix[end:end].as_matrix() / portvalue.ix[dates[i-1]:dates[i-1]].as_matrix()) - 1)[0]
        
        # alphas
        # apparently, cvxopt.matrix requires the input ndarray to be F_CONTIGUOUS which i discovered reading the C source code
        # F_CONTIGUOUS is found in ndarray.flags and is a boolean which ensure a Fortran-contiguous array
        # np.require forces that to be the case; this took me a really long time to figure out
        a0 = np.require(expected_excess_returns.ix[end:end].transpose().as_matrix(), dtype=np.float64, requirements=['F'])
        a = matrix(a0)
        
        if type == 'sample':
            S = matrix(cov.as_matrix())
        elif type == 'shrunk':
            # compute the shrunk covariance matrix, sigma
            sigma, shrinkage = port.get_shrunk_covariance_matrix(cov)
            S = matrix(sigma.as_matrix())
        else:
            raise ValueError('Type must be either of the two strings: sample or shrunk')
        
        # get the optimized weights
        # this is horribly naive because i'm only including the constaints provided in the example
        # i spent a considerable amount of time looking at the documentation, forums, and source
        # code trying to become comfortable with the package to no avail
        x = op.optimize(a, S)
        
        # optimized expected excess portfolio returns
        e_ = (x.T * y).sum()

        e.append(e_)

    return {
        'information_ratio': port.information_ratio(np.array([e])),
        'mean_excess_return': np.array([e]).mean(),
        'stdev_excess_return': np.array([e]).std()
    }

runs = 10
N = [30, 50, 100]

start = time.time()

for n in N:

    ir_sh = []; me_sh = []; se_sh = []; 
    ir_sa = []; me_sa = []; se_sa = []; 
    s = time.time()
    
    for i in xrange(runs):
        res = eval('sample', index=n)
        ir_sa.append(res['information_ratio'])
        me_sa.append(res['mean_excess_return'])
        se_sa.append(res['stdev_excess_return'])
        
        res = eval('shrunk', index=n)
        ir_sh.append(res['information_ratio'])
        me_sh.append(res['mean_excess_return'])
        se_sh.append(res['stdev_excess_return'])
        
        #print 'run #',i+1,'of',runs,'(',round(((i+1.0)/runs),2)*100.0,'% complete)'
    
    # sample cov matrix stats
    ir_sa = sum(ir_sa) / runs
    mer_sa = sum(me_sa) / runs
    msd_sa = sum(se_sa) / runs

    # shrunk cov matrix stats
    ir_sh = sum(ir_sh) / runs
    mer_sh = sum(me_sh) / runs
    msd_sh = sum(se_sh) / runs
    
    print '\t\tIR\t\tMean\tSD'
    print 'Sample\t%.4f\t%.4f\t%.4f\t' % (ir_sa, mer_sa, msd_sa)
    print 'Shrink\t%.4f\t%.4f\t%.4f\t' % (ir_sh, mer_sh, msd_sh)
    print 'computed in', round(time.time()-s, 2), 'seconds'
    print 'N=%.0f\tRuns=%.0f' % (n, runs)
    print

print 'total run', round((time.time()-start)/60.0, 2), 'minutes'