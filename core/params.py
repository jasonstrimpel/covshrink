from datetime import datetime

def get_portfolio_params():
    return {
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
            'max_position': 0.10,
            'target_gain': 0.03
        },
        'defaults': {
            'frequency': 'w'
        }
    }
    
def get_bench_params():
    return {
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