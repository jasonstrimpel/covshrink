# custom modules
import params
import portfolio

from datetime import datetime

# get the portfolio parameters
port_params = params.get_portfolio_params();
bench_params = params.get_bench_params();

# instantiate the porfolio object
port = portfolio.Portfolio(port_params, bench_params)


ticker = 'aapl'
start = datetime(2009, 1, 1)
end = datetime(2009, 12, 31)

shares = {
    'gs': 10,
    'c': 50,
    'jpm': 100,
    'tgt': 50,
    'wmt': 50,
    'f': 1000,
    'x': 25,
    'ibm': 20,
    'aapl': 5,
    'goog':10
}

value = port.get_portfolio_historic_values(shares=shares)
print value.describe()
