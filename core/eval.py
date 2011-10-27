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

def eval():

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

    # 
    dates = port.get_trading_dates()
    start = dates[0] + rollperiod
    end = dates[-1]

    delta = relativedelta.relativedelta(end, start)
    periods = (delta.years * 12) + delta.months

    portvalue = port.get_portfolio_historic_position_values().dropna()
    bench = port.get_benchmark_weights().T

    bench = pandas.DataFrame(np.tile(bench, (len(dates), 1)), index=dates, columns=bench.columns)
    returns = port.get_portfolio_historic_returns()

    expected_excess_returns = port.get_expected_excess_stock_returns()

    e = []

    for i in xrange(roll, periods+roll+1):
        
        # setup the dates to calculate returns for the covariance matrixes
        start = dates[i-roll]
        end = dates[i]
        
        #dataframe of positions over the roll period
        positions = portvalue.ix[start:end]
        
        # portfolio weights 
        total = portvalue.ix[start:end].sum(axis=1)
        port_weight = positions / total
        
        active_weights = port_weight - bench.ix[start:end]
        active_returns = returns.ix[start:end] * active_weights
        
        # compute the sample covariance matrix, cov
        #cov = port.get_covariance_matrix(returns.ix[start:end])
        cov = port.get_covariance_matrix(active_returns)
        # compute the shrunk covariance matrix, sigma
        sigma, shrinkage = port.get_shrunk_covariance_matrix(cov)
        
        # actual realized returns
        y = ((portvalue.ix[end:end].as_matrix() / portvalue.ix[dates[i-1]:dates[i-1]].as_matrix()) - 1)[0]
        
        # alphas
        # apparently, cvxopt.matrix requires the input ndarray to be F_CONTIGUOUS which i discovered reading the C source code
        # F_CONTIGUOUS is found in ndarray.flags and is a boolean which ensure a Fortran-contiguous array
        # np.require forces that to be the case; this took me a really long time to figure out
        a0 = np.require(expected_excess_returns.ix[end:end].transpose().as_matrix(), dtype=np.float64, requirements=['F'])
        a = matrix(a0)
        
        #S = matrix(sigma.as_matrix())
        S = matrix(cov.as_matrix())
        
        x = op.optimize(a, S)
        
        e_ = (x.T * y).sum()
        
        e.append(e_)

        '''
        print
        print 'i=',i,'date=',dates[i]
        print 'port return=',e_
        print '~*~*~*~*~*~*~*~'
        '''
        
    return {
        'information_ratio': port.information_ratio(np.array([e])),
        'mean_excess_return': np.array([e]).std(),
        'stdev_excess_return': np.array([e]).mean()
    }

s = time.time()

runs = 50.0
ir = []
for i in xrange(runs):
    res = eval()
    ir.append(res['information_ratio'])
    print 'run #',i+1,'of',runs,'(',round(((i+1.0)/runs),2)*100.0,'% complete)'

print 'mean ir = ', sum(ir) / runs, 'computed in', round(time.time()-s, 2), 'seconds'

