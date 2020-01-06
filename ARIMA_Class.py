import numpy as np
import json
import datetime
from scipy import stats
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import random
import statsmodels.api as sm
import seaborn as sb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from pytz import timezone
from PreProcessing_Class import PreProcessing

class DataMining:
	def __init__(self, start_year, end_year, start_month, end_month, run=False):
		start = datetime.datetime(start_year, start_month, 10, 0, 0, 0, tzinfo=timezone('Europe/Paris'))
		end = datetime.datetime(end_year, end_month, 31, 23, 59, 59, tzinfo=timezone('Europe/Paris'))
		startNY = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=timezone('US/Eastern'))
		endNY = datetime.datetime(end_year, end_month, 31, 23, 59, 59, tzinfo=timezone('US/Eastern'))

		self.year = start.year
		sb.set_style('darkgrid')
		prepro = PreProcessing(start, end, startNY, endNY)
		if run:
			prepro.dataset_creation()

		self.cities = prepro.cities
		self.files_opening()

	def files_opening(self):
		for filename in os.listdir('data'):
			if 'Torino' in filename:
				# print('Torino')
				Torino = pd.read_excel('data/'+filename)
				self.TO = self.data_filling(Torino, self.cities[0])
			elif 'Amsterdam' in filename:
				# print('Amsterdam')
				Amsterdam = pd.read_excel('data/'+filename)
				self.AM = self.data_filling(Amsterdam, self.cities[1])
			elif 'New York' in filename:
				# print('New York')
				NY = pd.read_excel('data/'+filename)
				self.NY = self.data_filling(NY, self.cities[2])

		self.dataframes = [self.TO, self.AM, self.NY]

	def data_filling(self, dataset, c):
		print('Filling Data for {}'.format(c))
		tmp_df = []
		isCompleted = 0
		while (not isCompleted):
			for row, col in dataset.iterrows():
				try:
					if dataset.loc[row, 'Hour'] - dataset.loc[row-1, 'Hour'] > 1: #if the current hour is not the before + 1, missing data
						for k in range(int(dataset.loc[row-1, 'Hour'])+1, dataset.loc[row, 'Hour']): #for loop from precedent hour to the current one
							tmp_row = [col['Day'], k,
										int((dataset.loc[row, 'Total'] + dataset.loc[row+1, 'Total'] + random.uniform(-5,5))/2)]
							print(tmp_row)
							tmp_df.append(tmp_row)
				except:
					print('No Index')
			if len(tmp_df) > 0:
				tmp_df = pd.DataFrame(tmp_df, columns=['Day', 'Hour', 'Total'])
				final_dataset = pd.concat([dataset.loc[:, 'Day':'Total'], tmp_df], ignore_index=True) #concatenate dataframes
				final_dataset = final_dataset.sort_values(['Day', 'Hour'])
				print('Filling Complete')
				final_dataset.to_excel('data/Data_{}.xlsx'.format(c))
				return final_dataset
			else:
				isCompleted = 1
				print('No new data inserted')
				return dataset

	def system_plotting(self, show=1, save=1):
		date = datetime.datetime(self.year, 1, 1)
		delta = int(self.dataframes[0].loc[0, 'Day']) - 1
		start = date + datetime.timedelta(days=delta)
		n = self.dataframes[0].loc[:, 'Day'].unique()
		# print(n)
		# exit()
		x_lab = []
		for i in range(len(n)):
			x_lab.append(start.strftime('%d %b'))
			start = start + datetime.timedelta(days=1)
		cnt = 0
		x_axis = [x*24 for x in range(len(n))]

		for i in self.dataframes:
			plt.figure(figsize=(12,4))
			plt.plot(list(i.loc[:,'Total']))
			plt.title(self.cities[cnt]+' Time Series')
			plt.ylabel('Number of Rentals')
			plt.xticks(x_axis, x_lab, rotation=45)
			if show:
				plt.show()
			if save:
				plt.savefig('plots/' + self.cities[cnt]+'_time_series.png')
			
			cnt+=1

	def plot_correlations(self, show=1, save=1, lags=48, names=['Torino', 'Amsterdam', 'New York City']):
		cnt = 0
		for df in self.dataframes:
			data = df.loc[:, 'Total']
			fig, ax = plt.subplots(2)
			plot_acf(data, ax=ax[0], lags=lags, title='ACF')
			plot_pacf(data, ax=ax[1], lags=lags, title='PACF')
			if show:
				plt.show()
			if save:
				plt.savefig('plots/ACF_PACF_{}.png'.format(names[cnt]))
			plt.close()
			cnt += 1
	
	def ts_preparation(self, df):
		df.loc[:, 'Total'] += np.random.random_sample(len(df))/10
		return df

	def model_fitting(self, train_size=24*275, append=0, sliding=0, start=650):
		# ps = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 22, 24]
		ps = [1,2,3,4]
		ds = [0, 1, 2]
		qs = [1, 2, 3, 4]

		with open('AllModels.json') as f:
			obj = json.loads(f.read()) #file in which all the models performances will be saved
			
		cnt = 0
		self.best_order = {} #object to save only the model with the lowest MAE
		for df in self.dataframes:
			df = df.loc[:, 'Total']
			train, test = df[0:train_size], df[train_size::]
			mae_min = 1e5
			for d in ds:
				for p in ps:
					for q in qs:
						history = [x for x in train]
						yhats = []
						print('ARIMA -> [{},{},{}]'.format(p,d,q))
						if append: #if append mode is chosen, the fitting will be evaluated forecasting only 1 element per time
							try:
								for t in range(len(test)):
									model = ARIMA(history, order=(p,d,q))
									model_fit = model.fit(disp=0)
									fc = model_fit.forecast()
									yhats.append(fc[0])
									history.append(fc[0])
									if sliding: #if sliding mode is chosen, the first element will be deleted from the training set
										history = history[1:]

								mae = mean_absolute_error(test, yhats)
								if mae < mae_min:
									print('Best Model for {} is --> [{},{},{}] with {} MAE'.format(self.cities[cnt],p,d,q,mae))
									self.best_order[self.cities[cnt]] = [p,d,q]
									mae_min = mae
							except:
								print('Parameters Error')
						else: #if append mode is not chosen, the system will evaluate all the testing element at once
							try:
								model = ARIMA(history, order=(p,d,q))
								model_fit = model.fit(maxiter=300, resolver='ncg', disp=0)
								fc, se, conf = model_fit.forecast(len(test.index))
								fc_s = pd.Series(fc, index=test.index)
								lowers = pd.Series(conf[:,0], index=test.index)
								uppers = pd.Series(conf[:,1], index=test.index)

								mae = mean_absolute_error(test, fc_s)
								tmp_obj = {
									'P':p,
									'D':d,
									'Q':q,
									'MAE':mae
								}
								obj[self.cities[cnt]].append(tmp_obj)

								print('Error: {}'.format(mae))
								with open('AllModels.json', 'w') as f:
									f.write(json.dumps(obj, indent=4))

								if mae < mae_min:
									print('Best Model for {} is --> [{},{},{}] with {} MAE'.format(self.cities[cnt],p,d,q,mae))
									self.best_order[self.cities[cnt]] = [p,d,q]
									mae_min = mae
							except:
								print('Parameters Error')
			cnt += 1

		with open('BestOrders.json', 'w+') as f:
			f.write(json.dumps(self.best_order, indent=4))
	
	def results_predicting(self, start, end):
		cnt = 0
		with open('BestOrders.json') as f:
			orders = json.loads(f.read())
		self.dataframes = [self.TO]
		for df in self.dataframes:
			# history = [x for x in df[0::]]
			# print(len(df.index))
			print(self.cities[cnt])
			order = orders[self.cities[cnt]]
			model_fit = ARIMA(df.loc[6000::,'Total'], order=order).fit(disp=0, tol=1e7)
			fc, _, conf = model_fit.forecast(10, alpha=0.05)
			index_of_fc = np.arange(len(df.index), end)

			fc_series = pd.Series(fc, index=index_of_fc)
			lower_series = pd.Series(conf[:, 0], index=index_of_fc)
			upper_series = pd.Series(conf[:, 1], index=index_of_fc)

			plt.plot(df.loc[start:, 'Total'])
			plt.plot(fc_series, color='darkgreen')
			plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

			plt.show()
			cnt+=1








