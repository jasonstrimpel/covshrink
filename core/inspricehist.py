# standard library imports
import time
import datetime
import urllib2
import os

#related third party imports
from matplotlib.cbook import iterable
from matplotlib import verbose, get_configdir
import matplotlib.mlab as mlab
try:
    from hashlib import md5
except ImportError:
    from md5 import md5 #Deprecated in 2.5
import tables

def _fetch_historical_yahoo(ticker, date1, date2, freq='w', proxy=None, cachename=None):
    """matplotlib's implementation, modified to provide proxy support and frequency
    Fetch historical data for ticker between date1 and date2.  date1 and
    date2 are date or datetime instances, or (year, month, day) sequences.

    Ex:
    fh = fetch_historical_yahoo('^GSPC', (2000, 1, 1), (2001, 12, 31))

    cachename is the name of the local file cache.  If None, will
    default to the md5 hash or the url (which incorporates the ticker
    and date range)

    a file handle is returned
    """

    ticker = ticker.upper()
    
    configdir = get_configdir()
    cachedir = os.path.join(configdir, 'finance.cache')

    if iterable(date1):
        d1 = (date1[1]-1, date1[2], date1[0])
    else:
        d1 = (date1.month-1, date1.day, date1.year)
    if iterable(date2):
        d2 = (date2[1]-1, date2[2], date2[0])
    else:
        d2 = (date2.month-1, date2.day, date2.year)

    if freq != 'd' or freq != 'w' or freq != 'm':
        freq = 'w'

    urlFmt = 'http://table.finance.yahoo.com/table.csv?a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&s=%s&y=0&g=%s&ignore=.csv'

    url =  urlFmt % (d1[0], d1[1], d1[2],
                     d2[0], d2[1], d2[2], ticker, freq)

    proxy = {"http": "http://proxy.jpmchase.net:8443"}

    if proxy:
        
        proxy_support = urllib2.ProxyHandler(proxy)
        opener = urllib2.build_opener(proxy_support)
        urllib2.install_opener(opener)

    if cachename is None:
        cachename = os.path.join(cachedir, md5(url).hexdigest())
    if os.path.exists(cachename):
        fh = file(cachename)
        verbose.report('Using cachefile %s for %s'%(cachename, ticker))
    else:
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        urlfh = urllib2.urlopen(url)

        fh = file(cachename, 'w')
        fh.write(urlfh.read())
        fh.close()
        verbose.report('Saved %s data to cache file %s'%(ticker, cachename))
        fh = file(cachename, 'r')

    return fh

def insert(ticker, start, end, frequency):
    """Inserts frequency price data for ticker from start to end
    
    Parameters
    ----------
    ticker : ticker symbol for which to insert data
    start : start date for data acquisition
    end : end date for data acquisition
    frequency : frequncy of data to acquire {d, w, m, y}
    
    Returns
    -------
    boolean : true on success, false on failure
    
    """
    # error checking here
    
    h5f = tables.openFile('price_data.h5', 'a')
    price_data = h5f.getNode('/price_data')
    
    fh = _fetch_historical_yahoo(ticker, start, end, frequency)
    # mlab.csv2rec converts lines in a csv to a record
    row = list(mlab.csv2rec(fh))
    fh.close()

    # loop through each item in the result set
    for item in row:

        k = list(item)

        # insert the frequency
        k.insert(1, frequency)

        newrow = price_data.row

        newrow['ticker'] = ticker
        newrow['frequency'] = k[1]
        newrow['date'] = time.mktime(time.strptime(k[0].strftime("%Y-%m-%d"), "%Y-%m-%d"))
        newrow['open'] = k[2]
        newrow['high'] = k[3]
        newrow['low'] = k[4]
        newrow['close'] = k[5]
        newrow['volume'] = k[6]
        newrow['adjustedClose'] = k[7]
        newrow['timestamp'] = time.time() 

        newrow.append()

    try:
        price_data.flush()
        h5f.close()
        return True

    except:
        return False