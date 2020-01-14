from ARIMA_Class import DataMining
from PreProcessing_Class import PreProcessing
import datetime

start_year = 2017
end_year = 2017
start_month = 10
end_month = 10
START = 6785
END = 6815

run = False
mining = DataMining(start_year, end_year, start_month, end_month, run=run)
# mining.system_plotting(show=1, save=0)
# mining.stationarity()
# mining.plot_correlations(show=0, save=1)
# mining.arima_training(append=True)
# mining.model_summary()
# mining.model_fitting()
# mining.error_plotting()
# mining.strategization()
mining.expanding()
# mining.results_predicting(START, END)
