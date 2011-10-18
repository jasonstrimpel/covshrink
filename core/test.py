# custom modules
import params
import portfolio
import optimize as op
import numpy as np
import pandas

from datetime import datetime
from datetime import date
import time

# get the portfolio parameters
port_params = params.get_portfolio_params();
bench_params = params.get_bench_params();

# instantiate the porfolio object
proxy = {"http": "http://proxy.jpmchase.net:8443"}
port = portfolio.Portfolio(port_params, bench_params, proxy=proxy)

rollperiod = 60

start = datetime(2005, 8, 1)
end = datetime(2011, 8, 23)

cov = port.get_covariance_matrix()
(sigma, shrinkage) = port.get_shrunk_covariance_matrix(cov)

dates = port.get_trading_dates()
portvalue = port.get_portfolio_historic_position_values()
bench = port.get_benchmark_weights().T

N = len(dates)

bench = pandas.DataFrame(np.tile(bench, (N, 1)), index=dates, columns=bench.columns)

for d in dates:
    positions = portvalue.ix[d:d]
    total = portvalue.ix[d:d].sum(axis=1)
    port_weight = positions / total
    active_weights = port_weight - bench.ix[d:d]
    
    