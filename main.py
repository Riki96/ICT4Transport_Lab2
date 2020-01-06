from ARIMA_Class import DataMining
from PreProcessing_Class import PreProcessing
import datetime

start_year = 2017
end_year = 2017
start_month = 10
end_month = 10
mining = DataMining(start_year, end_year, start_month, end_month)
# mining.system_plotting(show=1, save=0)
# mining.plot_correlations(show=0)
# mining.arima_training()
# mining.model_summary()
mining.model_fitting()
# mining.results_predicting(START, END)
