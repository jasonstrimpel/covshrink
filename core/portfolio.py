# standard modules
from datetime import datetime
import time

# application specific modules
import matplotlib.finance as fin
import numpy as np
import pandas
import tables

# custom modules
import yahoo
import inspricehist

__all__ = ['_get_historic_data', '_get_historic_returns', '_build_portfolio', 'get_benchmark_weights', 
                'get_active_weights', 'get_portfolio_weights', 'get_holding_period_returns', 'get_expected_stock_returns',
                'get_active_returns', 'get_expected_excess_stock_returns', 'get_covariance_matrix',
                'get_expected_benchmark_return', 'get_benchmark_variance', 'get_expected_portfolio_return',
                'get_portfolio_variance', 'get_expected_excess_portfolio_return', 'get_tracking_error_variance']

__version__ = '0.1'

__author__ = 'Jason Strimpel'

class Portfolio(object):

    def __init__(self, portfolio, benchmark):
        """Initializest the portfolio by creating and populating the data table
        
        Parameters
        ----------
        portfolio : a dictionary which contains all the information required to build the portfolio
            This includes:
                expected_returns : the expected return for each ticker; future enhancements allow
                    an alogrithm to build thid
                holding_periods : start and end date of holding for each position
                shares : number of shares held in each position
                constraints : constraints on the portfolio
                defaults : miscellaneous default values
        benchmark : a dictionary of weights of shares in the benchmark index
        
        Usage
        -------
        port = Portfolio(portfolio, bench)
        
        """
        # do error checking here
        
        holding_periods = portfolio['holding_periods']
        frequency = portfolio['defaults']['frequency']
        
        self._exp_ret = portfolio['expected_returns']
        self._hld_per = holding_periods
        self._shrs = portfolio['shares']
        self._freq = frequency
    
        self._bench_wts = benchmark
        
        # build the table for the data
        import createdailytable
        
        # load the data into the data table
        for symbol in holding_periods.keys():
            inspricehist.insert(symbol, holding_periods[symbol]['start'], holding_periods[symbol]['end'], frequency)

    def _get_historic_data(self, ticker, start, end):
        """Gets the open, high, low, close, volume, and adjusted close for the given ticker from start to end
        
        Parameters
        ----------
        (string) ticker : ticker symbol of the stock for which to get data
        (datetime) start : start date for historic data
        (datetime) end : end date for historic data
        
        Returns
        -------
        DataFrame : pandas.DataFrame object with open, high, low, close, and volume
        
        """
        frequency = self._freq
        
        h5f = tables.openFile('price_data.h5', 'r')
        price_data = h5f.getNode('/price_data')
        
        cols = tuple([n for n in price_data.colnames])
        colnames = cols

        start = time.mktime(time.strptime(start.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        end = time.mktime(time.strptime(end.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        
        condition = '(frequency == \'%s\') & (ticker == \'%s\') & (date >= start) & (date <= end)' % (frequency, ticker)
        res = price_data.readWhere(condition)
        
        h5f.close()
        
        cols = zip(*[row for row in res])
        data = dict(zip(colnames, cols))
        
        dates = pandas.Index([datetime.fromtimestamp(d) for d in data['date']])
        return pandas.DataMatrix(data, index=dates, dtype='float').sort(ascending=True)

    def _get_historic_returns(self, ticker, start, end, offset=1):
        """Calculates the offset-period return for the given tickers and combines in a DataFrame
        
        Parameters
        ----------
        (list) tickers : ticker symbols for which to include offset-period returns in the
            DataFrame
        (datetime) start : start date for historic data
        (datetime) end : end date for historic data
        (int) : offset for which to calculate returns
        
        Returns
        -------
        DataFrame : pandas.DataFrame object with offset-period returns for each ticker in tickers
        
        """
        prices = {}
        frame = self._get_historic_data(ticker, start, end)
        prices[ticker] = frame['adjustedClose']
        
        prices_frame = pandas.DataFrame(prices, dtype='float')
        
        return prices_frame / prices_frame.shift(offset) - 1

    def _build_portfolio(self, shares):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        positions = shares.keys()
        yh = yahoo.Yahoo(positions)
        
        prices = {}; portfolio = {}
        for position in positions:
            prices[position] = yh.get_LastTradePriceOnly(position)
        
        portfolio['shares'] = shares
        portfolio['price'] = prices

        return pandas.DataFrame(portfolio)

    def get_benchmark_weights(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        return pandas.DataFrame({
            'bench_weights': self._bench_wts
        })

    def get_active_weights(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        portfolio = self.get_portfolio_weights()
        bench = self.get_benchmark_weights()
        
        return pandas.DataFrame({
            'active_weights': portfolio['port_weights'] - bench['bench_weights']
        })

    def get_portfolio_weights(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        shares = self._shrs
        portfolio = self._build_portfolio(shares)
        
        mkt_val = portfolio['shares'] * portfolio['price']
        portfolio_val = mkt_val.sum()
        
        return pandas.DataFrame({
            'port_weights': mkt_val / portfolio_val
        })

    def get_holding_period_returns(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        holding_periods = self._hld_per
        positions = holding_periods.keys()
        
        holding_period_returns = {}
        for position in positions:
            prices = self._get_historic_data(position, holding_periods[position]['start'], holding_periods[position]['end'])
            holding_period_returns[position] = (prices['adjustedClose'][-1] / prices['adjustedClose'][0]) - 1
        
        return pandas.DataFrame({
            'holding_period_return': holding_period_returns
        })

    def get_expected_stock_returns(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        return pandas.DataFrame({
            'expected_returns': self._exp_ret
        })

    def get_active_returns(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        active_weights = self.get_active_weights()
        holding_period_returns = self.get_holding_period_returns()
        
        return pandas.DataFrame({
            'active_return': active_weights['active_weights'] * holding_period_returns['holding_period_return']
        })

    def get_expected_excess_stock_returns(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        expected_returns = self.get_expected_stock_returns()
        bench = self.get_benchmark_weights()
        
        return pandas.DataFrame({
            'expected_excess_returns': expected_returns['expected_returns'] - (bench['bench_weights'] * expected_returns['expected_returns'])
        })

    def get_covariance_matrix(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        holding_periods = self._hld_per
        positions = holding_periods.keys()
        
        historic_returns = {}
        for position in positions:
            returns = self._get_historic_returns(position, holding_periods[position]['start'], holding_periods[position]['end'])
            historic_returns[position] = returns[position]

        frame = pandas.DataFrame(historic_returns).dropna()

        cv = np.cov(frame,  rowvar=0)

        return pandas.DataFrame(cv, index=frame.columns, columns=frame.columns)


    def get_expected_benchmark_return(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        bench_weights = self.get_benchmark_weights()
        expected_portfolio_returns = self.get_expected_stock_returns()
        
        return pandas.DataFrame({
            'expected_benchmark_return': bench_weights['bench_weights'] * expected_portfolio_returns['expected_returns']
        })

    def get_benchmark_variance(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        bench_weights = self.get_benchmark_weights()
        cov_matrix = self.get_covariance_matrix()
        
        return pandas.DataFrame({
            'benchmark_variance': np.dot(bench_weights.T, np.dot(cov_matrix, bench_weights))
        })

    def get_expected_portfolio_return(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        portfolio_weights = self.get_portfolio_weights()
        expected_portfolio_returns = self.get_expected_stock_returns()
        
        return pandas.DataFrame({
            'expected_portfolio_return': portfolio_weights['port_weights'] * expected_portfolio_returns['expected_returns']
        })

    def get_portfolio_variance(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        port_weights = self.get_portfolio_weights()
        cov_matrix = self.get_covariance_matrix()
        
        return pandas.DataFrame({
            'portfolio_variance': np.dot(port_weights.T, np.dot(cov_matrix, port_weights))
        })

    def get_expected_excess_portfolio_return(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        active_weights = self.get_active_weights()
        expected_portfolio_returns = self.get_expected_stock_returns()
        
        return pandas.DataFrame({
            'expected_excess_portfolio_return': active_weights['active_weights'] * expected_portfolio_returns['expected_returns']
        })

    def get_tracking_error_variance(self):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        
        
        """
        active_weights = self.get_active_weights()
        cov_matrix = self.get_covariance_matrix()
        
        return pandas.DataFrame({
            'tracking_error_variance': np.dot(active_weights.T, np.dot(cov_matrix, active_weights))
        })

portfolio = {
    'expected_returns': {
        'gs': 0.08,
        'c': 0.05,
        'jpm': 0.15,
        'tgt': 0.08,
        'wmt': 0.10,
        'siri': 0.06,
        'x': 0.05,
        'ibm': 0.15,
        'aapl': 0.09,
        'goog': 0.15
    },
    'holding_periods': {
        'gs': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'c': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'jpm': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'tgt': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'wmt': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'siri': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'x': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'ibm': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'aapl': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)},
        'goog': {'start': datetime(2009, 8, 1), 'end': datetime(2011, 8, 23)}
    },
    'shares': {
        'gs': 10,
        'c': 50,
        'jpm': 100,
        'tgt': 50,
        'wmt': 50,
        'siri': 1000,
        'x': 25,
        'ibm': 20,
        'aapl': 5,
        'goog':10
    },
    'constraints': {
        'max_position': 0.10
    },
    'defaults': {
        'frequency': 'w'
    }
}

bench = {
    'gs': 0.08,
    'c': 0.05,
    'jpm': 0.15,
    'tgt': 0.08,
    'wmt': 0.12,
    'siri': 0.08,
    'x': 0.05,
    'ibm': 0.15,
    'aapl': 0.09,
    'goog': 0.15
}

ticker = 'gs'
start =  datetime(2011,1,1)
end = datetime(2011,1,31)

port = Portfolio(portfolio, bench)

#print port._get_historic_data(ticker, start, end)
#print port._get_historic_returns(ticker, start, end, offset=1)
#print port._build_portfolio(shares)
#print port.get_benchmark_weights()
#print port.get_active_weights()
#print port.get_portfolio_weights()
#print port.get_holding_period_returns()
#print port.get_expected_stock_returns()
#print port.get_active_returns()
#print port.get_expected_excess_stock_returns()
#print port.get_covariance_matrix().as_matrix()
#print port.get_expected_benchmark_return()
#print port.get_benchmark_variance()
#print port.get_expected_portfolio_return()
#print port.get_portfolio_variance()
#print port.get_expected_excess_portfolio_return()
#print port.get_tracking_error_variance()

