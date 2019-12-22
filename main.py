from ARIMA_Class import DataMining
from PreProcessing_Class import PreProcessing
import datetime



CITIES = ['Torino', 'Amsterdam', 'New York City']
# CITIES = ['Torino']
# start = datetime.datetime(2017, 10, 1, 0, 0, 0, tzinfo=timezone('Europe/Paris'))
# end = datetime.datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone('Europe/Paris'))

# startNY = datetime.datetime(2017, 10, 1, 0, 0, 0, tzinfo=timezone('US/Eastern'))
# endNY = datetime.datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone('US/Eastern'))

# prepro = PreProcessing()
# prepro.start()
# prepro.dataset_creation(start, end, startNY, endNY)
START = 650
END = 780
mining = DataMining()
# mining.system_plotting(show=1, save=0)
# mining.plot_correlations(show=0)
# mining.arima_training()
# mining.model_summary()
mining.model_fitting()
# mining.results_predicting(START, END)
