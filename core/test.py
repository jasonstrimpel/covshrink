# standard modules
from datetime import datetime
from datetime import date
from dateutil import relativedelta
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
    
    cov = port.get_covariance_matrix(returns.ix[start:end])
    
    sigma, shrinkage = port.get_shrunk_covariance_matrix(cov)
    y = ((portvalue.ix[dates[i]:dates[i]].as_matrix() / portvalue.ix[dates[i-1]:dates[i-1]].as_matrix()) - 1)[0]
    
    a = matrix(port.get_expected_excess_stock_returns().as_matrix())
    S = matrix(sigma.as_matrix())
    
    x = op.optimize(a, S)
    
    e_ = (x.T * y).sum()
    
    e.append(e_)

    
    '''
    print
    print 'i=',i,'date=',dates[i]
    print 'port return=',e.sum()
    print 'ir=',
    print '~*~*~*~*~*~*~*~'
    '''

print 'ir=',port.information_ratio(np.array([e]))
print 'e std=',np.array([e]).std()
print 'd mean=',np.array([e]).mean()