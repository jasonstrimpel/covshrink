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

# internal (private) methods
print port._get_historic_data(ticker, start, end)
print port._get_historic_returns(ticker, start, end, offset=1)
print port._build_portfolio(shares)

#  public methods

print port.get_portfolio_historic_returns() # red
print port.get_portfolio_historic_position_values()
print port.get_portfolio_historic_values()

print port.get_benchmark_weights()
print port.get_active_weights()
print port.get_portfolio_weights()
print port.get_holding_period_returns()
print port.get_expected_stock_returns()
print port.get_active_returns()
print port.get_expected_excess_stock_returns()
print port.get_covariance_matrix()
print port.get_expected_benchmark_return()
print port.get_benchmark_variance()
print port.get_expected_portfolio_return()
print port.get_portfolio_variance()
print port.get_expected_excess_portfolio_return()
print port.get_tracking_error_variance()