#related third party imports
import tables

h5f = tables.openFile('price_data.h5', 'w')

description = {
    "ticker": tables.StringCol(itemsize=6, dflt='', pos=1),
    "frequency": tables.StringCol(itemsize=1, dflt='d', pos=2),
    "date": tables.Time32Col(dflt=0.00, pos=3),
    "open": tables.Float32Col(dflt=0.00, pos=4),
    "high": tables.Float32Col(dflt=0.00, pos=5),
    "low": tables.Float32Col(dflt=0.00, pos=6),
    "close": tables.Float32Col(dflt=0.00, pos=7),
    "volume": tables.Int64Col(dflt=0.00, pos=8),
    "adjustedClose": tables.Float32Col(dflt=0.00, pos=9),
    "timestamp": tables.Time64Col(dflt=0.00, pos=10)
}

table = h5f.createTable('/', 'price_data', description)

h5f.close()