import urllib2
import json

class Yahoo(object):
    
    def __init__(self, ticker_list):
        
        proxy ={"http": "http://proxy.jpmchase.net:8443"}
        
        if proxy:
            
            proxy_support = urllib2.ProxyHandler(proxy)
            opener = urllib2.build_opener(proxy_support)
            urllib2.install_opener(opener)
        
        tickers = '%22%2C%22'.join(ticker_list).upper()
        
        url = 'http://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20yahoo.finance.quotes%20where%20symbol%20in%20(%22'+tickers+'%22)&format=json&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback='
        req = urllib2.Request(url)

        try:
            response = urllib2.urlopen(req)
            
        except urllib2.URLError, e:

            if hasattr(e, 'reason'):
                print 'We failed to reach a server with reason:', e.reason
                print 'The URL passed was:', url
                print 'The tickers passed were:', tickers
                print 'The response from Yahoo was:', e.read()
                print
            elif hasattr(e, 'code'):
                print 'The server couldn\'t fulfill the request with error code:', e.code
                print 'The URL passed was:', url
                print 'The tickers passed were:', tickers
                print 'The response from Yahoo was:', e.read()
                print

        result = json.loads(response.read())
        data = result['query']['results']['quote']

        self._data = dict(zip(ticker_list, data))

    def get_symbol(self, symbol):
        get = self._data
        return get[symbol]['symbol']

    def get_Ask(self, symbol):
        get = self._data
        return get[symbol]['Ask']

    def get_AverageDailyVolume(self, symbol):
        get = self._data
        return get[symbol]['AverageDailyVolume']

    def get_Bid(self, symbol):
        get = self._data
        return get[symbol]['Bid']

    def get_AskRealtime(self, symbol):
        get = self._data
        return get[symbol]['AskRealtime']

    def get_BidRealtime(self, symbol):
        get = self._data
        return get[symbol]['BidRealtime']

    def get_BookValue(self, symbol):
        get = self._data
        return get[symbol]['BookValue']

    def get_Change_PercentChange(self, symbol):
        get = self._data
        return get[symbol]['Change_PercentChange']

    def get_Change(self, symbol):
        get = self._data
        return get[symbol]['Change']

    def get_Commission(self, symbol):
        get = self._data
        return get[symbol]['Commission']

    def get_ChangeRealtime(self, symbol):
        get = self._data
        return get[symbol]['ChangeRealtime']

    def get_AfterHoursChangeRealtime(self, symbol):
        get = self._data
        return get[symbol]['AfterHoursChangeRealtime']

    def get_DividendShare(self, symbol):
        get = self._data
        return get[symbol]['DividendShare']

    def get_LastTradeDate(self, symbol):
        get = self._data
        return get[symbol]['LastTradeDate']

    def get_TradeDate(self, symbol):
        get = self._data
        return get[symbol]['TradeDate']

    def get_EarningsShare(self, symbol):
        get = self._data
        return get[symbol]['EarningsShare']

    def get_ErrorIndicationreturnedforsymbolchangedinvalid(self, symbol):
        get = self._data
        return get[symbol]['ErrorIndicationreturnedforsymbolchangedinvalid']

    def get_EPSEstimateCurrentYear(self, symbol):
        get = self._data
        return get[symbol]['EPSEstimateCurrentYear']

    def get_EPSEstimateNextYear(self, symbol):
        get = self._data
        return get[symbol]['EPSEstimateNextYear']

    def get_EPSEstimateNextQuarter(self, symbol):
        get = self._data
        return get[symbol]['EPSEstimateNextQuarter']

    def get_DaysLow(self, symbol):
        
        ask = self.get_AskRealtime(symbol)
        
        get = self._data
        days_low = get[symbol]['DaysLow']
        
        if ask < days_low:
            return ask
        else:
            return days_low

    def get_DaysHigh(self, symbol):
        
        ask = self.get_AskRealtime(symbol)
        
        get = self._data
        days_high = get[symbol]['DaysHigh']
        
        if ask > days_high:
            return ask
        else:
            return days_high

    def get_YearLow(self, symbol):
        get = self._data
        return get[symbol]['YearLow']

    def get_YearHigh(self, symbol):
        get = self._data
        return get[symbol]['YearHigh']

    def get_HoldingsGainPercent(self, symbol):
        get = self._data
        return get[symbol]['HoldingsGainPercent']

    def get_AnnualizedGain(self, symbol):
        get = self._data
        return get[symbol]['AnnualizedGain']

    def get_HoldingsGain(self, symbol):
        get = self._data
        return get[symbol]['HoldingsGain']

    def get_HoldingsGainPercentRealtime(self, symbol):
        get = self._data
        return get[symbol]['HoldingsGainPercentRealtime']

    def get_HoldingsGainRealtime(self, symbol):
        get = self._data
        return get[symbol]['HoldingsGainRealtime']

    def get_MoreInfo(self, symbol):
        get = self._data
        return get[symbol]['MoreInfo']

    def get_OrderBookRealtime(self, symbol):
        get = self._data
        return get[symbol]['OrderBookRealtime']

    def get_MarketCapitalization(self, symbol):
        get = self._data
        return get[symbol]['MarketCapitalization']

    def get_MarketCapRealtime(self, symbol):
        get = self._data
        return get[symbol]['MarketCapRealtime']

    def get_EBITDA(self, symbol):
        get = self._data
        return get[symbol]['EBITDA']

    def get_ChangeFromYearLow(self, symbol):
        get = self._data
        return get[symbol]['ChangeFromYearLow']

    def get_PercentChangeFromYearLow(self, symbol):
        get = self._data
        return get[symbol]['PercentChangeFromYearLow']

    def get_LastTradeRealtimeWithTime(self, symbol):
        get = self._data
        return get[symbol]['LastTradeRealtimeWithTime']

    def get_ChangePercentRealtime(self, symbol):
        get = self._data
        return get[symbol]['ChangePercentRealtime']

    def get_ChangeFromYearHigh(self, symbol):
        get = self._data
        return get[symbol]['ChangeFromYearHigh']

    def get_PercentChangeFromYearHigh(self, symbol):
        '''
            Note the incorrect spelling, this is yahoo
        '''
        get = self._data
        return get[symbol]['PercebtChangeFromYearHigh']

    def get_LastTradeWithTime(self, symbol):
        get = self._data
        return get[symbol]['LastTradeWithTime']

    def get_LastTradePriceOnly(self, symbol):
        get = self._data
        return float(get[symbol]['LastTradePriceOnly'])

    def get_HighLimit(self, symbol):
        get = self._data
        return get[symbol]['HighLimit']

    def get_LowLimit(self, symbol):
        get = self._data
        return get[symbol]['LowLimit']

    def get_DaysRange(self, symbol):
        get = self._data
        return get[symbol]['DaysRange']

    def get_DaysRangeRealtime(self, symbol):
        get = self._data
        return get[symbol]['DaysRangeRealtime']

    def get_FiftydayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['FiftydayMovingAverage']

    def get_TwoHundreddayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['TwoHundreddayMovingAverage']

    def get_ChangeFromTwoHundreddayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['ChangeFromTwoHundreddayMovingAverage']

    def get_PercentChangeFromTwoHundreddayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['PercentChangeFromTwoHundreddayMovingAverage']

    def get_ChangeFromFiftydayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['ChangeFromFiftydayMovingAverage']

    def get_PercentChangeFromFiftydayMovingAverage(self, symbol):
        get = self._data
        return get[symbol]['PercentChangeFromFiftydayMovingAverage']

    def get_Name(self, symbol):
        get = self._data
        return get[symbol]['Name']

    def get_Notes(self, symbol):
        get = self._data
        return get[symbol]['Notes']

    def get_Open(self, symbol):
        get = self._data
        return get[symbol]['Open']

    def get_PreviousClose(self, symbol):
        get = self._data
        return get[symbol]['PreviousClose']

    def get_PricePaid(self, symbol):
        get = self._data
        return get[symbol]['PricePaid']

    def get_ChangeinPercent(self, symbol):
        get = self._data
        return get[symbol]['ChangeinPercent']

    def get_PriceSales(self, symbol):
        get = self._data
        return get[symbol]['PriceSales']

    def get_PriceBook(self, symbol):
        get = self._data
        return get[symbol]['PriceBook']

    def get_ExDividendDate(self, symbol):
        get = self._data
        return get[symbol]['ExDividendDate']

    def get_PERatio(self, symbol):
        get = self._data
        return get[symbol]['PERatio']

    def get_DividendPayDate(self, symbol):
        get = self._data
        return get[symbol]['DividendPayDate']

    def get_PERatioRealtime(self, symbol):
        get = self._data
        return get[symbol]['PERatioRealtime']

    def get_PEGRatio(self, symbol):
        get = self._data
        return get[symbol]['PEGRatio']

    def get_PriceEPSEstimateCurrentYear(self, symbol):
        get = self._data
        return get[symbol]['PriceEPSEstimateCurrentYear']

    def get_PriceEPSEstimateNextYear(self, symbol):
        get = self._data
        return get[symbol]['PriceEPSEstimateNextYear']

    def get_Symbol(self, symbol):
        get = self._data
        return get[symbol]['Symbol']

    def get_SharesOwned(self, symbol):
        get = self._data
        return get[symbol]['SharesOwned']

    def get_ShortRatio(self, symbol):
        get = self._data
        return get[symbol]['ShortRatio']

    def get_LastTradeTime(self, symbol):
        get = self._data
        return get[symbol]['LastTradeTime']

    def get_TickerTrend(self, symbol):
        get = self._data
        return get[symbol]['TickerTrend']

    def get_OneyrTargetPrice(self, symbol):
        get = self._data
        return get[symbol]['OneyrTargetPrice']

    def get_Volume(self, symbol):
        get = self._data
        return get[symbol]['Volume']

    def get_HoldingsValue(self, symbol):
        get = self._data
        return get[symbol]['HoldingsValue']

    def get_HoldingsValueRealtime(self, symbol):
        get = self._data
        return get[symbol]['HoldingsValueRealtime']

    def get_YearRange(self, symbol):
        get = self._data
        return get[symbol]['YearRange']

    def get_DaysValueChange(self, symbol):
        get = self._data
        return get[symbol]['DaysValueChange']

    def get_DaysValueChangeRealtime(self, symbol):
        get = self._data
        return get[symbol]['DaysValueChangeRealtime']

    def get_StockExchange(self, symbol):
        get = self._data
        return get[symbol]['StockExchange']

    def get_DividendYield(self, symbol):
        get = self._data
        return get[symbol]['DividendYield']

    def get_PercentChange(self, symbol):
        get = self._data
        return get[symbol]['PercentChange']