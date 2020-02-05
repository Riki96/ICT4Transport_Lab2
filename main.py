from ARIMA_Class import DataMining
from PreProcessing_Class import PreProcessing
import datetime

if __name__ == '__main__':
	start_year = 2017
	end_year = 2017
	start_month = 10
	end_month = 10

	run = False
	mining = DataMining(start_year, end_year, start_month, end_month, run=run)
	# mining.model_fitting()
	