from datetime import datetime
from datetime import date
from dateutil import relativedelta
import numpy as np
from itertools import islice

start = datetime(1990, 2, 1)
end = datetime(2001, 12, 31)

def get_portfolio_params(index=30):

    expected_returns = {
        'BMY': 0.03,
        'C': 0.03,
        'CA': 0.03,
        'CAG': 0.03,
        'CAH': 0.03,
        'CAT': 0.03,
        'CB': 0.03,
        'CCE': 0.03,
        'CCL': 0.03,
        'CEG': 0.03,
        'CI': 0.03,
        'CL': 0.03,
        'CLF': 0.03,
        'CLX': 0.03,
        'CMCSA': 0.03,
        'CMI': 0.03,
        'CMS': 0.03,
        'CNP': 0.03,
        'COP': 0.03,
        'COST': 0.03,
        'CPB': 0.03,
        'CSC': 0.03,
        'CSX': 0.03,
        'CTL': 0.03,
        'CVS': 0.03,
        'CVX': 0.03,
        'D': 0.03,
        'DD': 0.03,
        'DE': 0.03,
        'DELL': 0.03,
        'DHR': 0.03,
        'DIS': 0.03,
        'DNB': 0.03,
        'DOV': 0.03,
        'DOW': 0.03,
        'DTE': 0.03,
        'DUK': 0.03,
        'ECL': 0.03,
        'ED': 0.03,
        'EFX': 0.03,
        'EIX': 0.03,
        'EMC': 0.03,
        'EMR': 0.03,
        'EOG': 0.03,
        'EQT': 0.03,
        'ETN': 0.03,
        'ETR': 0.03,
        'EXC': 0.03,
        'F': 0.03,
        'FDO': 0.03,
        'FDX': 0.03,
        'FMC': 0.03,
        'FRX': 0.03,
        'GAS': 0.03,
        'GCI': 0.03,
        'GD': 0.03,
        'GE': 0.03,
        'GIS': 0.03,
        'GLW': 0.03,
        'GPC': 0.03,
        'GPS': 0.03,
        'GR': 0.03,
        'GT': 0.03,
        'GWW': 0.03,
        'HAL': 0.03,
        'HAR': 0.03,
        'HAS': 0.03,
        #'HCP': 0.03,
        'HD': 0.03,
        'HES': 0.03,
        'HNZ': 0.03,
        'HOG': 0.03,
        'HON': 0.03,
        #'HOT': 0.03,
        'HP': 0.03,
        'HPQ': 0.03,
        'HRB': 0.03,
        'HRS': 0.03,
        'HST': 0.03,
        'HSY': 0.03,
        'HUM': 0.03,
        'IBM': 0.03,
        'IFF': 0.03,
        'INTC': 0.03,
        'IP': 0.03,
        'IPG': 0.03,
        'IR': 0.03,
        'ITW': 0.03,
        'JCI': 0.03,
        'JCP': 0.03,
        'JNJ': 0.03,
        'JPM': 0.03,
        'JWN': 0.03,
        'K': 0.03,
        'KEY': 0.03,
        'KMB': 0.03,
        'KO': 0.03,
        'KR': 0.03,
        'L': 0.03,
        'LEG': 0.03,
        'AA': 0.03,
        'AAPL': 0.03,
        'AXP': 0.03,
        'BA': 0.03,
        'BAC': 0.03,
        'BP': 0.03,
        'CAT': 0.03,
        'CVX': 0.03,
        'DD': 0.03,
        'DIS': 0.03,
        'GE': 0.03,
        'HD': 0.03,
        #'HPQ': 0.03, doesn't have a 2/2002 data point which is throwing everything off
        'IBM': 0.03,
        'INTC': 0.03,
        'JNJ': 0.03,
        'JPM': 0.03,
        'KO': 0.03,
        'MCD': 0.03,
        'MMM': 0.03,
        'MRK': 0.03,
        'MSFT': 0.03,
        'PFE': 0.03,
        'PG': 0.03,
        'T': 0.03,
        'TGT': 0.03,
        'UTX': 0.03,
        'VZ': 0.03,
        'WMT': 0.03,
        'XOM': 0.03
    }
    holding_periods = {
        'BMY': {'start': start, 'end': end},
        'C': {'start': start, 'end': end},
        'CA': {'start': start, 'end': end},
        'CAG': {'start': start, 'end': end},
        'CAH': {'start': start, 'end': end},
        'CAT': {'start': start, 'end': end},
        'CB': {'start': start, 'end': end},
        'CCE': {'start': start, 'end': end},
        'CCL': {'start': start, 'end': end},
        'CEG': {'start': start, 'end': end},
        'CI': {'start': start, 'end': end},
        'CL': {'start': start, 'end': end},
        'CLF': {'start': start, 'end': end},
        'CLX': {'start': start, 'end': end},
        'CMCSA': {'start': start, 'end': end},
        'CMI': {'start': start, 'end': end},
        'CMS': {'start': start, 'end': end},
        'CNP': {'start': start, 'end': end},
        'COP': {'start': start, 'end': end},
        'COST': {'start': start, 'end': end},
        'CPB': {'start': start, 'end': end},
        'CSC': {'start': start, 'end': end},
        'CSX': {'start': start, 'end': end},
        'CTL': {'start': start, 'end': end},
        'CVS': {'start': start, 'end': end},
        'CVX': {'start': start, 'end': end},
        'D': {'start': start, 'end': end},
        'DD': {'start': start, 'end': end},
        'DE': {'start': start, 'end': end},
        'DELL': {'start': start, 'end': end},
        'DHR': {'start': start, 'end': end},
        'DIS': {'start': start, 'end': end},
        'DNB': {'start': start, 'end': end},
        'DOV': {'start': start, 'end': end},
        'DOW': {'start': start, 'end': end},
        'DTE': {'start': start, 'end': end},
        'DUK': {'start': start, 'end': end},
        'ECL': {'start': start, 'end': end},
        'ED': {'start': start, 'end': end},
        'EFX': {'start': start, 'end': end},
        'EIX': {'start': start, 'end': end},
        'EMC': {'start': start, 'end': end},
        'EMR': {'start': start, 'end': end},
        'EOG': {'start': start, 'end': end},
        'EQT': {'start': start, 'end': end},
        'ETN': {'start': start, 'end': end},
        'ETR': {'start': start, 'end': end},
        'EXC': {'start': start, 'end': end},
        'F': {'start': start, 'end': end},
        'FDO': {'start': start, 'end': end},
        'FDX': {'start': start, 'end': end},
        'FMC': {'start': start, 'end': end},
        'FRX': {'start': start, 'end': end},
        'GAS': {'start': start, 'end': end},
        'GCI': {'start': start, 'end': end},
        'GD': {'start': start, 'end': end},
        'GE': {'start': start, 'end': end},
        'GIS': {'start': start, 'end': end},
        'GLW': {'start': start, 'end': end},
        'GPC': {'start': start, 'end': end},
        'GPS': {'start': start, 'end': end},
        'GR': {'start': start, 'end': end},
        'GT': {'start': start, 'end': end},
        'GWW': {'start': start, 'end': end},
        'HAL': {'start': start, 'end': end},
        'HAR': {'start': start, 'end': end},
        'HAS': {'start': start, 'end': end},
        #'HCP': {'start': start, 'end': end},
        'HD': {'start': start, 'end': end},
        'HES': {'start': start, 'end': end},
        'HNZ': {'start': start, 'end': end},
        'HOG': {'start': start, 'end': end},
        'HON': {'start': start, 'end': end},
        #'HOT': {'start': start, 'end': end},
        'HP': {'start': start, 'end': end},
        'HPQ': {'start': start, 'end': end},
        'HRB': {'start': start, 'end': end},
        'HRS': {'start': start, 'end': end},
        'HST': {'start': start, 'end': end},
        'HSY': {'start': start, 'end': end},
        'HUM': {'start': start, 'end': end},
        'IBM': {'start': start, 'end': end},
        'IFF': {'start': start, 'end': end},
        'INTC': {'start': start, 'end': end},
        'IP': {'start': start, 'end': end},
        'IPG': {'start': start, 'end': end},
        'IR': {'start': start, 'end': end},
        'ITW': {'start': start, 'end': end},
        'JCI': {'start': start, 'end': end},
        'JCP': {'start': start, 'end': end},
        'JNJ': {'start': start, 'end': end},
        'JPM': {'start': start, 'end': end},
        'JWN': {'start': start, 'end': end},
        'K': {'start': start, 'end': end},
        'KEY': {'start': start, 'end': end},
        'KMB': {'start': start, 'end': end},
        'KO': {'start': start, 'end': end},
        'KR': {'start': start, 'end': end},
        'L': {'start': start, 'end': end},
        'LEG': {'start': start, 'end': end},
        'AA': {'start': start, 'end': end},
        'AAPL': {'start': start, 'end': end},
        'AXP': {'start': start, 'end': end},
        'BA': {'start': start, 'end': end},
        'BAC': {'start': start, 'end': end},
        'BP': {'start': start, 'end': end},
        'CAT': {'start': start, 'end': end},
        'CVX': {'start': start, 'end': end},
        'DD': {'start': start, 'end': end},
        'DIS': {'start': start, 'end': end},
        'GE': {'start': start, 'end': end},
        'HD': {'start': start, 'end': end},
        #'HPQ': {'start': start, 'end': end}, doesn't have a 2/2002 data point which is throwing everything off
        'IBM': {'start': start, 'end': end},
        'INTC': {'start': start, 'end': end},
        'JNJ': {'start': start, 'end': end},
        'JPM': {'start': start, 'end': end},
        'KO': {'start': start, 'end': end},
        'MCD': {'start': start, 'end': end},
        'MMM': {'start': start, 'end': end},
        'MRK': {'start': start, 'end': end},
        'MSFT': {'start': start, 'end': end},
        'PFE': {'start': start, 'end': end},
        'PG': {'start': start, 'end': end},
        'T': {'start': start, 'end': end},
        'TGT': {'start': start, 'end': end},
        'UTX': {'start': start, 'end': end},
        'VZ': {'start': start, 'end': end},
        'WMT': {'start': start, 'end': end},
        'XOM': {'start': start, 'end': end}
    }
    shares = {
        'BMY': 19,
        'C': 76,
        'CA': 52,
        'CAG': 4,
        'CAH': 90,
        'CAT': 54,
        'CB': 8,
        'CCE': 31,
        'CCL': 59,
        'CEG': 73,
        'CI': 63,
        'CL': 43,
        'CLF': 63,
        'CLX': 3,
        'CMCSA': 68,
        'CMI': 48,
        'CMS': 50,
        'CNP': 77,
        'COP': 83,
        'COST': 60,
        'CPB': 17,
        'CSC': 64,
        'CSX': 52,
        'CTL': 72,
        'CVS': 28,
        'CVX': 31,
        'D': 98,
        'DD': 63,
        'DE': 29,
        'DELL': 31,
        'DHR': 76,
        'DIS': 90,
        'DNB': 78,
        'DOV': 2,
        'DOW': 51,
        'DTE': 16,
        'DUK': 72,
        'ECL': 66,
        'ED': 25,
        'EFX': 74,
        'EIX': 28,
        'EMC': 98,
        'EMR': 22,
        'EOG': 7,
        'EQT': 31,
        'ETN': 45,
        'ETR': 44,
        'EXC': 21,
        'F': 4,
        'FDO': 34,
        'FDX': 88,
        'FMC': 23,
        'FRX': 49,
        'GAS': 66,
        'GCI': 89,
        'GD': 69,
        'GE': 84,
        'GIS': 43,
        'GLW': 46,
        'GPC': 92,
        'GPS': 92,
        'GR': 32,
        'GT': 33,
        'GWW': 46,
        'HAL': 78,
        'HAR': 11,
        'HAS': 44,
        #'HCP': 0,
        'HD': 87,
        'HES': 2,
        'HNZ': 78,
        'HOG': 88,
        'HON': 19,
        #'HOT': 56,
        'HP': 49,
        'HPQ': 7,
        'HRB': 77,
        'HRS': 95,
        'HST': 63,
        'HSY': 24,
        'HUM': 80,
        'IBM': 66,
        'IFF': 52,
        'INTC': 15,
        'IP': 67,
        'IPG': 72,
        'IR': 2,
        'ITW': 85,
        'JCI': 61,
        'JCP': 92,
        'JNJ': 18,
        'JPM': 9,
        'JWN': 63,
        'K': 41,
        'KEY': 41,
        'KMB': 28,
        'KO': 48,
        'KR': 9,
        'L': 92,
        'LEG': 59,
        'AA': 55,
        'AAPL': 15,
        'AXP': 17,
        'BA': 56,
        'BAC': 110,
        'BP': 10,
        'CAT': 50,
        'CVX': 40,
        'DD': 20,
        'DIS': 15,
        'GE': 17,
        'HD': 66,
        #'HPQ': 11, # doesn't have a 2/2002 data point which is throwing everything off
        'IBM': 45,
        'INTC': 75,
        'JNJ': 60,
        'JPM': 37,
        'KO': 15,
        'MCD': 55,
        'MMM': 25,
        'MRK': 35,
        'MSFT': 27,
        'PFE': 65,
        'PG': 45,
        'T': 95,
        'TGT': 85,
        'UTX': 22,
        'VZ': 11,
        'WMT': 66,
        'XOM': 25
    }
    constraints = {
        'min_position': 0.05,
        'max_position': 0.15,
        'target_gain': 0.03
    }
    defaults = {
        'frequency': 'm',
        'start': start,
        'end': end
    }

    return {
        'expected_returns': dict(list(islice(expected_returns.iteritems(), index))),
        'holding_periods':dict(list(islice(holding_periods.iteritems(), index))),
        'shares': dict(list(islice(shares.iteritems(), index))),
        'constraints': constraints,
        'defaults': defaults
    }