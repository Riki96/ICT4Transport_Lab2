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
import traceback

class DataMining:
	def __init__(self, start_year, end_year, start_month, end_month, run=False):
		start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=timezone('Europe/Paris'))
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

	def files_opening(self, filling=False):
		for filename in os.listdir('data'):
			if 'Torino' in filename:
				# print('Torino')
				self.TO = pd.read_excel('data/'+filename)
				
			elif 'Amsterdam' in filename:
				# print('Amsterdam')
				self.AM = pd.read_excel('data/'+filename)
				
			elif 'New York' in filename:
				# print('New York')
				self.NY = pd.read_excel('data/'+filename)
		# print(self.TO)

		if filling:
			self.TO = self.data_filling(self.TO, self.cities[0])
			self.AM = self.data_filling(self.AM, self.cities[1])
			self.NY = self.data_filling(self.NY, self.cities[2])

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
	
	def stationarity(self):
		for df in self.dataframes:
			plt.figure()
			df['Total'].hist()
			one, two, three = np.split(df['Total'].sample(frac=1),[int(.25*len(df['Total'])),int(.75*len(df['Total']))])
			m1, m2, m3 = one.mean(), two.mean(), three.mean()
			v1, v2, v3 = one.var(), two.var(), three.var()

			print(m1,m2,m3)
			print(v1,v2,v3)

			adftest = sm.tsa.stattools.adfuller(df['Total'])
			print(adftest)

			plt.figure()
			pd.plotting.lag_plot(df['Total'])
			plt.show()

	def mean_relative_error(self, test, prediction):
		sum_ = 0
		for i in range(len(test)):
			diff = abs(test[i] - prediction[i])/abs(test[i])
			sum_ += diff

		mre = sum_ / len(test)
		return mre*100

	def mean_relative_error2(self, test, prediction):
		return abs(test-prediction)/abs(test)*100

	def model_fitting(self, train_size=24*7, test_size=24*7):
		ps = [1,2,3,4,5,6]
		ds = [0,1]
		qs = [1,2,3,4,5,6]
		# ps = [2]
		# ds = [0]
		# qs = [2]
			
		obj = []
		obj.append({
			"Torino":[]
			})
		obj.append({
			"Amsterdam":[]
			})
		obj.append({
			"New York City":[]
			})
		error_to = []
		error_am = []
		error_ny = []
		cnt = 0

		for df in self.dataframes:
			df = df['Total'].to_numpy()
			train, test = df[0:train_size], df[train_size:train_size+test_size]
			mre_min = 1e5
			for d in ds:
				for p in ps:
					for q in qs:
						print('ARIMA -> [{},{},{}]'.format(p,d,q))
						try:
							fc_ = []
							history = [x for x in train]
							for t in range(test_size):
								model = ARIMA(history, order=(p,d,q))
								model_fit = model.fit(disp=0)
								# fc, se, conf = model_fit.forecast(len(test.index))
								fc, se, conf = model_fit.forecast()
								fc_.append(fc[0])
								history.append(test[t])

							mre = self.mean_relative_error(test, fc_)
							tmp_obj = {
								'P':p,
								'D':d,
								'Q':q,
								'MRE':mre
							}
							obj[cnt][self.cities[cnt]].append(tmp_obj)
							print('Error: {}'.format(mre))
							with open('AllModels_2.json', 'w') as f:
								f.write(json.dumps(obj, indent=4))

							if mre < mre_min:
								print('Best Model for {} is --> [{},{},{}] with {} MRE'.format(self.cities[cnt],p,d,q,mre))
								# self.best_order[self.cities[cnt]] = [p,d,q]
								mre_min = mre
						except:
							traceback.print_exc()
							tmp_obj = {
								'P':p,
								'D':d,
								'Q':q,
								'MRE':'Impossible to fit'
							}
							obj[cnt][self.cities[cnt]].append(tmp_obj)

			cnt += 1

	def error_plotting(self, show=1):
		with open('AllModels_2.json') as f:
			models = json.loads(f.read())

		to_models = models['Torino']
		am_models = models['Amsterdam']
		ny_models = models['New York City']

		plt.figure()
		y = []
		for i in to_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])
		plt.plot(y)
		y = []
		for i in am_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])
		plt.plot(y, color='red')
		y = []
		for i in ny_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])

		plt.plot(y, color='green')
		if show:
			plt.show()

	def best_models(self):
		with open('AllModels_2.json') as f:
			models = json.loads(f.read())

		bestTO = {}
		bestAM = {}
		bestNY = {}

		for i in models:
			keyz = i.keys()
			# print(keyz)
			best = 100
			if 'Torino' in keyz:
				for j in i['Torino']:
					try:
						if j['MRE'] - best < -0.1 and j['MRE'] > 0:
							best = j['MRE']
							bestTO['P'] = j['P']
							bestTO['Q'] = j['Q']
							bestTO['D'] = j['D']
							bestTO['MRE'] = j['MRE']
					except:
						pass
			elif 'Amsterdam' in keyz:
				for j in i['Amsterdam']:
					try:
						if j['MRE'] - best < -0.1 and j['MRE'] > 0:
							best = j['MRE']
							bestAM['P'] = j['P']
							bestAM['Q'] = j['Q']
							bestAM['D'] = j['D']
							bestAM['MRE'] = j['MRE']
					except:
						pass

			elif 'New York City' in keyz:
				for j in i['New York City']:
					try:
						if j['MRE'] - best < -0.1 and j['MRE'] > 0:
							best = j['MRE']
							bestNY['P'] = j['P']
							bestNY['Q'] = j['Q']
							bestNY['D'] = j['D']
							bestNY['MRE'] = j['MRE']
					except:
						pass

		print(bestTO)
		print(bestAM)
		print(bestNY)

		return bestTO, bestAM, bestNY

	def plot_data_vs_predict(self, train_size=24*7, test_size=24*7):
		best_models = self.best_models()
		cnt = 0
		for df in self.dataframes:
			df = df['Total'].to_numpy()
			train, test = df[0:train_size], df[train_size:train_size+test_size]

			p,d,q = best_models[cnt]['P'], best_models[cnt]['D'], best_models[cnt]['Q']
			print(p,d,q)
			fc_ = []
			history = [x for x in train]
			for t in range(test_size):
				model = ARIMA(history, order=(p,d,q)).fit(disp=0)
				fc,conf,se = model.forecast()
				fc_.append(fc[0])
				history.append(test[t])

			plt.figure()
			plt.plot(test)
			plt.plot(fc_, label='Model [%d,%d,%d]'%(p,d,q))
			plt.show()

			cnt += 1

	def prepare_models(self):
		bestTO, bestAM, bestNY = self.best_models()
		
		p_to, d_to, q_to = bestTO['P'], bestTO['D'], bestTO['Q']
		p_am, d_am, q_am = bestAM['P'], bestAM['D'], bestAM['Q']
		p_ny, d_ny, q_ny = bestNY['P'], bestNY['D'], bestNY['Q']
		mre_to, mre_am, mre_ny = bestTO['MRE'], bestAM['MRE'], bestNY['MRE']
		ps = [p_to, p_am, p_ny]
		qs = [q_to, q_am, q_ny]
		ds = [d_to, d_am, d_ny]
		mres = [mre_to, mre_am, mre_ny]

		return ps, ds, qs		
	
	def expanding(self, train_start=120, test_size=24*7):
		cnt = 0
		bestTO, bestAM, bestNY = self.best_models()
		
		p_to, d_to, q_to = bestTO['P'], bestTO['D'], bestTO['Q']
		p_am, d_am, q_am = bestAM['P'], bestAM['D'], bestAM['Q']
		p_ny, d_ny, q_ny = bestNY['P'], bestNY['D'], bestNY['Q']
		mre_to, mre_am, mre_ny = bestTO['MRE'], bestAM['MRE'], bestNY['MRE']
		ps = [p_to, p_am, p_ny]
		qs = [q_to, q_am, q_ny]
		ds = [d_to, d_am, d_ny]
		mres = [mre_to, mre_am, mre_ny]

		#expanding
		cnt = 0
		for df in self.dataframes:
			df = df['Total'].to_numpy()
			train = df[0:train_start] #first 7 days of data
			errs = []
			try:
				history = [x for x in train]
				# for i in range(len(df) - train_start):
				for i in range(24*7):
					train = df[0:i+train_start]
					history = [x for x in train]
					fc_ = []
					for j in range(test_size):
						print(j)
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast()
						fc_.append(fc)
						history.append(df[i+train_start+j])
					mre = self.mean_relative_error(df[i+train_start:i+train_start+test_size], fc_)
					print(mre)
					errs.append(mre)
			except:
				traceback.print_exc()

			plt.figure()
			plt.title('Expanding Strategy [W=1 Hour] - {}'.format(self.cities[cnt]))
			plt.ylabel('Mean Relative Error')
			plt.plot(errs)
			plt.savefig('plots/ExpandingStrategy{}'.format(self.cities[cnt].strip()))
			cnt += 1
			exit()


	def sliding(self, test_size=24*7):
		bestTO, bestAM, bestNY = self.best_models()
		
		p_to, d_to, q_to = bestTO['P'], bestTO['D'], bestTO['Q']
		p_am, d_am, q_am = bestAM['P'], bestAM['D'], bestAM['Q']
		p_ny, d_ny, q_ny = bestNY['P'], bestNY['D'], bestNY['Q']
		mre_to, mre_am, mre_ny = bestTO['MRE'], bestAM['MRE'], bestNY['MRE']
		ps = [p_to, p_am, p_ny]
		qs = [q_to, q_am, q_ny]
		ds = [d_to, d_am, d_ny]
		mres = [mre_to, mre_am, mre_ny]
		cnt = 0
		sliding_size = [24*5, 24*7, 24*14, 24*21]

		for w in sliding_size:
			for df in self.dataframes:
				train = df[0:sliding_size]
				history = [x for x in train]

				for i in range(24*7):
					fc_ = []
					for j in range(test_size):
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast()
						fc_.append(fc)
						history.append(df[sliding_size+i+j])
						
			plt.figure()
			plt.title('Sliding Strategy [W=1 Hour] - {}'.format(self.cities[cnt]))
			plt.ylabel('Mean Relative Error')
			plt.plot(errs)
			plt.savefig('plots/SlidingStrategy{}'.format(self.cities[cnt].strip()))
			cnt += 1

	def horizon(self, train_size=24*14):
		ps, ds, qs = self.prepare_models()

		df = self.AM
		df = df['Total'].to_numpy()
		train, test = df[0:train_size], df[train_size::]
		

		ws = [i for i in range(3,25,3)]
		# print(w)
		for w in ws:
			history = [x for x in train]
			errs = []
			for i in range(0,len(test),w):
				print(round(i/len(test),2))
				model = ARIMA(history, order=(ps[1], ds[1], qs[1])).fit(disp=-1)
				fc,conf,se = model.forecast(w)
				mre = self.mean_relative_error(test[i:i+w], fc)
				print(mre)
				# history.append(test[i:i+w])
				errs.append(mre)
				try:
					for j in range(w):
						history.append(test[i+j])
				except:
					pass

			plt.figure()
			plt.plot(errs)
			plt.title('MRE with Window {}'.format(w))
			plt.ylabel('MRE')
			plt.savefig('plots/HorizonWindow{}'.format(w))







