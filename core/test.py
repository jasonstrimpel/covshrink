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


def jump_by_month(start_date, end_date, month_step=1): 
    current_date = start_date 
    while current_date < end_date: 
        yield current_date 
        carry, new_month = divmod(current_date.month - 1 + month_step, 12)
        new_month += 1 
        current_date = current_date.replace(year=current_date.year + carry, month=new_month) 

# get the portfolio parameters
port_params = params.get_portfolio_params();
bench_params = params.get_bench_params();

# instantiate the porfolio object
proxy = {"http": "http://proxy.jpmchase.net:8443"}
port = portfolio.Portfolio(port_params, bench_params, proxy=proxy)

roll = 36
rollperiod = relativedelta.relativedelta(months=roll)

#start = datetime(2005, 8, 1) + rollperiod

start = datetime(1990, 1, 1)
end = datetime(2000, 1, 1)

#cov = port.get_covariance_matrix()
#(sigma, shrinkage) = port.get_shrunk_covariance_matrix(cov)

dates = port.get_trading_dates()
portvalue = port.get_portfolio_historic_position_values()
bench = port.get_benchmark_weights().T

N = len(dates)

bench = pandas.DataFrame(np.tile(bench, (N, 1)), index=dates, columns=bench.columns)
returns = port.get_portfolio_historic_returns()

for i in range(len(dates)-roll):

    start = dates[i]
    end = dates[i] + rollperiod

    positions = portvalue.ix[start:end]

    total = portvalue.ix[start:end].sum(axis=1)
    port_weight = positions / total

    active_weights = port_weight - bench.ix[start:end]
    active_returns = returns.ix[start:end] * active_weights

    cov = port.get_covariance_matrix(active_returns)
    sigma, shrinkage = port.get_shrunk_covariance_matrix(cov)
    
    pbar = matrix(port.get_expected_excess_portfolio_return().as_matrix())
    S = matrix(cov.as_matrix())
    
    op_weights = op.optimize(pbar, S)
    
    print [i for i in op_weights]
    
