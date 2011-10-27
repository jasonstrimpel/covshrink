# standard modules
from datetime import datetime
from datetime import date
from dateutil import relativedelta
import math
import time

# custom modules
import params
import portfolio
import optimize as op
import numpy as np
import pandas
from cvxopt import matrix

'''
def jump_by_month(start_date, end_date, month_step=1): 
    current_date = start_date 
    while current_date < end_date: 
        yield current_date 
        carry, new_month = divmod(current_date.month - 1 + month_step, 12)
        new_month += 1 
        current_date = current_date.replace(year=current_date.year + carry, month=new_month) 
'''

def eval(type):

    # get the portfolio parameters
    port_params = params.get_portfolio_params()
    bench_params = params.get_bench_params()

    # instantiate the porfolio object
    port = portfolio.Portfolio(port_params, bench_params, proxy={"http": "http://proxy.jpmchase.net:8443"})

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
        '''
        #dataframe of positions over the roll period
        positions = portvalue.ix[start:end]
        
        # portfolio weights 
        total = portvalue.ix[start:end].sum(axis=1)
        port_weight = positions / total
        
        active_weights = port_weight - bench.ix[start:end]
        active_returns = returns.ix[start:end] * active_weights
        '''
        active_returns = active.ix[start:end]
        
        # compute the sample covariance matrix, cov of active returns
        cov = port.get_covariance_matrix(active_returns)
        # should it be absolute returns?
        #cov = port.get_covariance_matrix(returns.ix[start:end])
        
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
        'mean_excess_return': np.array([e]).std(),
        'stdev_excess_return': np.array([e]).mean()
    }



runs = 50
types = ['sample', 'shrunk']

for type in types:
    
    ir = []; me = []; se = []; 
    s = time.time()

    for i in xrange(runs):
        res = eval(type)
        ir.append(res['information_ratio'])
        me.append(res['mean_excess_return'])
        se.append(res['stdev_excess_return'])
        
        print 'run #',i+1,'of',runs,'(',round(((i+1.0)/runs),2)*100.0,'% complete)'
    
    print 'type:', type
    print 'mean ir = ', sum(ir) / runs
    print 'mean expected returns = ', sum(me) / runs
    print 'stdev expected returns = ', sum(se) / runs
    print 'computed in', round((time.time()-s)/60.0, 2), 'minutes'
    print