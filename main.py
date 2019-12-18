from ARIMA_Class import *
from pytz import timezone


CITIES = ['Torino', 'Amsterdam', 'New York City']
# CITIES = ['Torino']
start = datetime.datetime(2017, 10, 1, 0, 0, 0, tzinfo=timezone('Europe/Paris'))
end = datetime.datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone('Europe/Paris'))

startNY = datetime.datetime(2017, 10, 1, 0, 0, 0, tzinfo=timezone('US/Eastern'))
endNY = datetime.datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone('US/Eastern'))

prepro = PreProcessing()
prepro.dataset_creation(CITIES, start, end, startNY, endNY)

mining = DataMining()
# Torino, _, _ = mining.files_opening()
# mining.plot_correlations(plot=False)
# mining.data_filling(Torino)
mining.system_plotting(CITIES, start)
