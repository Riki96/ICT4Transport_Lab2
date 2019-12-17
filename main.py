from ARIMA_Class import *
from pytz import timezone


CITIES = ['Torino', 'Amsterdam', 'New York City']
start = datetime.datetime(2017, 10, 1)
end = datetime.datetime(2017, 10, 31, 23, 59, 59)

startNY = datetime.datetime(2017, 10, 1, tzinfo=timezone('US/Eastern'))
endNY = datetime.datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone('US/Eastern'))

# prepro = PreProcessing()
# prepro.dataset_creation(CITIES, start, end, startNY, endNY)

mining = DataMining()
mining.acf()
