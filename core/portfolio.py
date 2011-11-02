# standard modules
from math import sqrt
from math import log
from datetime import datetime
import time

# application specific modules
import matplotlib.finance as fin
import numpy as np
import pandas
import tables

# custom modules
import yahoo
import inspricehist as ph
import createdailytable

__all__ = ["_get_historic_data", "_get_historic_returns", "get_portfolio_historic_returns", 
                "get_portfolio_historic_position_values", "get_portfolio_historic_values", "get_benchmark_weights", 
                "get_benchmark_returns", "get_active_weights", "get_portfolio_weights", "get_expected_stock_returns", 
                "get_active_returns", "get_expected_excess_stock_returns", "get_covariance_matrix", 
                "get_shrunk_covariance_matrix", "get_expected_benchmark_return", "get_expected_portfolio_return", 
                "get_portfolio_size", "get_trading_dates", "information_ratio"]

__version__ = '0.1'

__author__ = 'Jason Strimpel'

class Portfolio(object):

    def __init__(self, portfolio, start=None, end=None, proxy=None):
        """Initializes the portfolio by creating and populating the data table. Goes out to Yahoo and gets historic 
            data using a Matplotlib method modified to accept a proxy and frequency of data
        
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
        
        Usage
        -------
        import params
        
        port_params = params.get_portfolio_params();
        port = Portfolio(port_params)
        
        # internal (private) methods
        print port._get_historic_data(ticker, start, end)
        print port._get_historic_returns(ticker, start, end, offset=1)

        #  public methods
        print port.get_portfolio_historic_returns()
        print port.get_portfolio_historic_position_values(shares=None)
        print port.get_portfolio_historic_values(shares=None)
        print port.get_benchmark_weights()
        print port.get_benchmark_returns()
        print port.get_active_weights()
        print port.get_portfolio_weights(shares=None)
        print port.get_expected_stock_returns()
        print port.get_active_returns()
        print port.get_expected_excess_stock_returns()
        print port.get_covariance_matrix(historic_returns)
        print port.get_shrunk_covariance_matrix(x, shrink=None)
        print port.get_expected_benchmark_return()
        print port.get_expected_portfolio_return()
        print port.get_portfolio_size()
        print port.get_trading_dates()
        print port.information_ratio(historic_returns)

        """
        # if optional start and end params are not provided, use the default values
        if start is not None:
            if type(start) == str or type(start) == datetime:
                start = time.mktime(time.strptime(start.strftime("%Y-%m-%d"), "%Y-%m-%d"))
            else:
                raise ValueError('Start date must be string (yyyy-mm-dd) or datetime object')
        else:
            start = portfolio['defaults']['start']
        
        if end is not None:
            if type(end) == str or type(end) == datetime:
                end = time.mktime(time.strptime(end.strftime("%Y-%m-%d"), "%Y-%m-%d"))
            else:
                raise ValueError('End date must be string (yyyy-mm-dd) or datetime object')
        else:
            start = portfolio['defaults']['end']
        
        holding_periods = portfolio['holding_periods']
        frequency = portfolio['defaults']['frequency']
        
        self._exp_ret = portfolio['expected_returns']
        self._hld_per = holding_periods
        self._shrs = portfolio['shares']
        self._freq = frequency
        self._start = start
        self._end = end
        
        # for those of use behind a proxy
        self._proxy = proxy
        
        # build the table for the data
        createdailytable.reset_table()
        
        phobj = ph.InsertPriceHist(self._proxy)
        
        # load the data into the data table
        for symbol in holding_periods.keys():
            phobj.insert(symbol, holding_periods[symbol]['start'], holding_periods[symbol]['end'], frequency)

    def _get_historic_data(self, ticker, start, end):
        """Translates the data stored in the pytables table containing the price data to a pandas.DataFrame
        
        Parameters
        ----------
        ticker : ticker symbol for which to get data
        start : datetime object or string object representing the date for which to begin gathering data
        end : datetime object or string object representing the date for which to end gathering data
        
        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the historic data for ticker from start to end
        
        """
        frequency = self._freq
        
        h5f = tables.openFile('price_data.h5', 'r')
        price_data = h5f.getNode('/price_data')
        
        cols = tuple([n for n in price_data.colnames])
        colnames = cols

        if type(start) == str or type(start) == datetime:
            start = time.mktime(time.strptime(start.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        else:
            raise ValueError('Start date must be string (yyyy-mm-dd) or datetime object')
        
        if type(end) == str or type(end) == datetime:
            end = time.mktime(time.strptime(end.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        else:
            raise ValueError('End date must be string (yyyy-mm-dd) or datetime object')

        condition = '(frequency == \'%s\') & (ticker == \'%s\') & (date >= start) & (date <= end)' % (frequency, ticker)
        res = price_data.readWhere(condition)
        
        h5f.close()
        
        cols = zip(*[row for row in res])
        data = dict(zip(colnames, cols))
        
        dates = pandas.Index([datetime.fromtimestamp(d) for d in data['date']])
        return pandas.DataFrame(data, index=dates, dtype='float').sort(ascending=True)

    def _get_historic_returns(self, ticker, start, end, offset=1):
        """Gets historic data for ticker from start to end and computes offset period returns
        
        Parameters
        ----------
        ticker : ticker symbol for which to get data
        start : datetime object or string object representing the date for which to begin gathering data
        end : datetime object or string object representing the date for which to end gathering data
        offset : periods for which to calculate the returns
        
        Returns
        -------
        pandas.Series : pandas.Series with return data for ticker
        
        """

        prices = self._get_historic_data(ticker, start, end)
        return pandas.Series(prices['adjustedClose'] / prices['adjustedClose'].shift(offset) - 1)

    def get_portfolio_historic_returns(self):
        """Computes the historic returns of the portfolio

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the historic returns of the portfolio
        
        """
        shares = self._shrs
        positions = shares.keys()
        periods = self._hld_per 
        
        returns = {}
        for position in positions:
            returns[position] = self._get_historic_returns(position, periods[position]['start'], periods[position]['end'])
        
        return pandas.DataFrame(returns)

    def get_portfolio_historic_position_values(self, shares=None):
        """Computes the historic value of the positions in the portfolio
        
        Parameters
        ----------
        shares : this should be a matrix of historic share quantity, currently not used
        
        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the values of the portfolio constituents
        
        """
        if shares is None:
            shares = self._shrs
        
        positions = shares.keys()
        periods = self._hld_per 

        prices = {}; portfolio = {}
        for position in positions:
            frame = self._get_historic_data(position, periods[position]['start'], periods[position]['end'])
            prices[position] = frame['adjustedClose'] * shares[position]
 
        return pandas.DataFrame(prices)

    def get_portfolio_historic_values(self, shares=None):
        """Computes the historic value of the entire portfolio
        
        Parameters
        ----------
        shares: optional parameter shares overrides the default share quantity passed through the
            parameters
        
        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the historic portfolio values
        
        """
        if shares is None:
            shares = self._shrs
        
        positions = shares.keys()
        periods = self._hld_per 
        
        values = {}
        for position in positions:
            prices = self._get_historic_data(position, periods[position]['start'], periods[position]['end'])
            values[position] = prices['adjustedClose'] * shares[position]
 
        portfolio = pandas.DataFrame(values).sum(axis=1)
 
        return pandas.Series(portfolio)

    def get_benchmark_weights(self):
        """Returns the weights of the benchmark constituents
            Could be arrays of actual weights or computed weights
        
        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the weights of the benchmark constituents
        
        """
        positions = self._shrs.keys()
        
        shares = {}
        for ticker in positions:
            shares[ticker] = 1

        portvalue = self.get_portfolio_historic_position_values(shares)
        total = portvalue.sum(axis=1)
        
        bench_weight = portvalue / total

        return bench_weight

    def get_benchmark_returns(self):
        """Computes the returns on the constituents of the benchmark (same as portfolio only with 1 share)

        Returns
        -------
        pandas.dataFrame : pandas.dataFrame with returns of the benchmark constituents
        
        """
        positions = self._shrs.keys()
        
        shares = {}
        for ticker in positions:
            shares[ticker] = 1
        
        return self.get_portfolio_historic_position_values(shares) / self.get_portfolio_historic_position_values(shares).shift(1) - 1

    def get_active_weights(self):
        """Computes the active portfolio weights

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame with active weights for each portfolio component
        
        """
        portfolio = self.get_portfolio_weights()
        bench = self.get_benchmark_weights()
        
        return portfolio - bench
    
    def get_portfolio_weights(self, shares=None):
        """Computes the weights of the portfolio constituents including share holdings

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame containing the portfolio weights of the portfolio
            constituents
        
        """
        if shares is None:
            shares = self._shrs
        
        portvalue = self.get_portfolio_historic_position_values().dropna()
        port_weight = portvalue / portvalue.sum(axis=1)
        
        return port_weight
    
    def get_expected_stock_returns(self):
        """Returns the expected stock returns as defined in the input parameters
            This could be modified to return expected stock returns based on an alpha model
        
        Returns
        -------
        pandas.DataFrame :  pandas.DataFrame which contains the expected stock returns on the portfolio
            constituents
        
        """
        return pandas.DataFrame({
            'expected_returns': self._exp_ret
        })
    
    def get_active_returns(self):
        """Computes the active returns on the portfolio constituents
        
        Returns
        -------
        pandas.DataFrame :  pandas.DataFrame which contains the active stock returns on the portfolio
            constituents
        
        """
        active_weights = self.get_active_weights()
        returns = self.get_portfolio_historic_returns()

        return active_weights * returns

    
    def get_expected_excess_stock_returns(self):
        """Computes the expected excess stock returns
            The authors build the expected excess returns by adding random noise to the realized excess returns, 
            using a one-period lognormal model of returns (mu + sigma * randn(N, i)) where N is the number of 
            stocks and i is the number of periods. mu was assumed 0.03 and sigma 0.05. The authors then build
            alpha in a such a way that the unconstrained annualized ex-ante information ratio (IR) is 1.5. This 
            procedure is described in Appendix C of "Honey, I Shrunk the Sample Covariance Matrix".

        Returns
        -------
        pandas.DataFrame :  pandas.DataFrame which contains the expected excess stock returns on the portfolio
            constituents
        
        """
        freq = self._freq
        
        if freq == 'y':
            f = 1
        elif freq == 'm':
            f = 12
        elif freq == 'w':
            f = 52
        elif freq == 'd':
            f = 252
        
        expected_returns = self.get_expected_stock_returns()
        bench_weights = self.get_benchmark_weights()
        
        portvalue = self.get_portfolio_historic_position_values()
        total = portvalue.sum(axis=1)
        port_weights = portvalue / total
        
        excess_returns = self.get_portfolio_historic_returns() * (port_weights - bench_weights)
        
        # one period lognormal model for noise
        # assumes ln(S / S_0) = m + s * randn
        # assumes mean = 0.03 and sigma = 0.05
        # could enhance by using estimates generated by CAPM
        N = self.get_portfolio_size()
        m = 0.03
        s = 0.05
        noise = m + s * np.random.randn(np.shape(excess_returns)[0], np.shape(excess_returns)[1])
        raw = excess_returns + noise
        
        # IR ~ IC * sqrt(breadth)
        # breadth = freq * N where N is the benchmark size
        # the authors fix the IR at 1.5 so IC = 1.5 / sqrt(freq * N)
        ic = 1.5 / sqrt(f * N)
        
        # step 2.
        score = (raw - raw.mean()) / raw.std()
        alpha = (excess_returns.std() * ic * score).dropna()

        return alpha

    def get_covariance_matrix(self, historic_returns):
        """Computes a sample covariance matrix given historic returns
        
        Parameters
        ----------
        historic_returns : an NxM pandas.DataFrame or np.array of historic returns with
            N assets and M periods
        
        Returns
        -------
        pandas.DataFrame : returns an NxN pandas.DataFrame covariance matrix
        
        """
        frame = pandas.DataFrame(historic_returns).dropna()
        return pandas.DataFrame(np.cov(frame,  rowvar=0), index=frame.columns, columns=frame.columns)

    def get_shrunk_covariance_matrix(self, x, shrink=None):
        """Computes a covariance matrix that is "shrunk" towards a structured estimator. Code
            borrows heavily from the MATLAB implementation available by the authors online.
        
        Parameters
        ----------
        x : N x N sample covariance matrix of stock returns
        shrink : given shrinkage intensity factor; if none, code calculates
        
        Returns
        -------
        tuple : pandas.DataFrame which contains the shrunk covariance matrix
                : float shrinkage intensity factor
        
        """
        if x is None:
            raise ValueError('No covariance matrix defined')
        
        if type(x) == pandas.core.frame.DataFrame:
            cov = x.as_matrix()
        elif type(x) == np.ndarray:
            cov = x
        else:
            raise ValueError('Covariance matrix passed must be numpy.ndarray or pandas.DataFrame')
        
        if shrink is not None:
            shrinkage = shrink

        index = x.index
        columns = x.columns
        
        [t, n] = np.shape(cov)
        meanx = cov.mean(axis=0)
        cov = cov - np.tile(meanx, (t, 1))
        
        sample = (1.0 / t) * np.dot(cov.T, cov)
        
        var = np.diag(sample)
        sqrtvar = np.sqrt(var)

        a = np.tile(sqrtvar, (n, 1))
        rho = (sum(sum(sample / (a * a.T))) - n) / (n*(n-1))
        
        prior = rho * (a * a.T)
        prior[np.eye(t, n)==1] = var
        
        # Frobenius-norm of matrix cov, sqrt(sum(diag(dot(cov.T, cov))))
        # have to research this
        c = np.linalg.norm(sample-prior, 'fro')**2
        y = cov**2.0
        p = np.dot((1.0 / t), sum(sum(np.dot(y.T, y))))-sum(sum(sample**2.0))
        rdiag = np.dot((1.0 / t), sum(sum(y**2.0))) - sum(var**2.0)
        v = np.dot((cov**3.0).T, cov) / t - ((var*sample).T)
        v[np.eye(t, n)==1] = 0.0
        roff = sum(sum(v * (a  / a.T)))
        r = rdiag + np.dot(rho, roff)
        
        # compute shrinkage constant
        k = (p - r) / c
        shrinkage = max(0.0, min(1.0, k/t))
        sigma = np.dot(shrinkage, prior) + np.dot((1 - shrinkage), sample)
        
        return pandas.DataFrame(sigma, index=index, columns=columns), shrinkage

    def get_expected_benchmark_return(self):
        """Computes the expected return on the benchmark

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame which contains the expected returns on the benchmark
            constituents
        
        """
        bench_weights = self.get_benchmark_weights()
        expected_portfolio_returns = self.get_expected_stock_returns()
        
        return pandas.DataFrame({
            'expected_benchmark_return': bench_weights['bench_weights'] * expected_portfolio_returns['expected_returns']
        })

    def get_expected_portfolio_return(self):
        """Computes the expected return on the portfolio

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame which contains the expected returns on the portfolio
            constituents
        
        """
        portfolio_weights = self.get_portfolio_weights()
        expected_portfolio_returns = self.get_expected_stock_returns()
        
        return pandas.DataFrame({
            'expected_portfolio_return': portfolio_weights['port_weights'] * expected_portfolio_returns['expected_returns']
        })

    def get_portfolio_size(self):
        """Computes the number of assets in the portfolio
        
        Returns
        -------
        integer : number of assets in the portfolio
        
        """
        holding_periods = self._hld_per
        positions = holding_periods.keys()
        return len(positions)
    
    def get_trading_dates(self):
        """Returns the dates included for which there are historic returns
        
        Returns
        -------
        list : returns a list of datetime objects corresponding to the dates for which there are historic returns
        
        """
        return self.get_portfolio_historic_returns().index
    
    def information_ratio(self, historic_returns):
        """Computes the information ratio
            IR ~ IC * sqrt(breadth)
            In our case, breadth is the number of bets per year. Bets per year is the frequency
            provided. By default, monthly.
        
        Parameters
        ----------
        historic_returns : an NxM pandas.DataFrame or np.array of historic returns with
            N assets and M periods
        
        Returns
        -------
        float : information ratio as defined by Grinold and Kahn
        
        """
        freq = self._freq
        
        if freq == 'y':
            f = 1
        elif freq == 'm':
            f = 12
        elif freq == 'w':
            f = 52
        elif freq == 'd':
            f = 252
        
        mean = historic_returns.mean()
        stdev = historic_returns.std()
        
        return (sqrt(f) * mean) / stdev