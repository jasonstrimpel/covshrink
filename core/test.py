# standard modules
from datetime import datetime
from datetime import date
from dateutil import relativedelta
from itertools import islice
import time

# custom modules
import params2
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
port_params = params2.get_portfolio_params(index=125)
# instantiate the porfolio object
port = portfolio.Portfolio(port_params, proxy={"http": "http://proxy.jpmchase.net:8443"})


print port.get_portfolio_historic_position_values().to_string()

'''
bench = port.get_benchmark_weights().ix[datetime(1990,02,01)]
print
print
print
portvalue = port.get_portfolio_historic_position_values().dropna()
port_weight = portvalue / portvalue.sum(axis=1)

p = port_weight.ix[datetime(1990,02,01)]

#print p - bench

active = port.get_active_returns()
print 
print 
print active.ix[datetime(1990,03,01)]
'''