from ARIMA_Class import DataMining
from PreProcessing_Class import PreProcessing
import datetime

start_year = 2017
end_year = 2017
start_month = 8
end_month = 10
run = False
START = 6785
END = 6815
mining = DataMining(start_year, end_year, start_month, end_month, run=run)
# mining.system_plotting(show=1, save=0)
# mining.plot_correlations(show=0, save=1)
# mining.arima_training()
# mining.model_summary()
# mining.model_fitting()
mining.results_predicting(START, END)
