from PreProcessing_Class import PreProcessing
import os
import json
import numpy as np
import datetime
from scipy import stats
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import random
import statsmodels.api as sm
import seaborn as sb
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from pytz import timezone
import traceback

class DataMining:
	def __init__(self, start_year, end_year, start_month, end_month, run=False):
		sb.set_style('darkgrid')
		#Amsterdam and Torino have the same timezone. For NewYork instead we need to use another one
		start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=timezone('Europe/Paris'))
		end = datetime.datetime(end_year, end_month, 31, 23, 59, 59, tzinfo=timezone('Europe/Paris'))
		startNY = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=timezone('US/Eastern'))
		endNY = datetime.datetime(end_year, end_month, 31, 23, 59, 59, tzinfo=timezone('US/Eastern'))

		self.year = start.year
		prepro = PreProcessing(start, end, startNY, endNY)
		if run:
			prepro.dataset_creation()

		self.cities = prepro.cities
		self.files_opening()

	def files_opening(self, filling=False):
		for filename in os.listdir('data'):
			if 'Torino' in filename:
				self.TO = pd.read_excel('data/'+filename)
				
			elif 'Amsterdam' in filename:
				self.AM = pd.read_excel('data/'+filename)
				
			elif 'NewYork' in filename:
				self.NY = pd.read_excel('data/'+filename)

		if filling:
			self.TO = self.data_filling(self.TO, self.cities[0])
			self.AM = self.data_filling(self.AM, self.cities[1])
			self.NY = self.data_filling(self.NY, self.cities[2])

		#create list of dataframes to use further in the code
		self.dataframes = [self.TO, self.AM, self.NY]

	def data_filling(self, dataset, c):
		#replace missing data with the average between the previous and the following number of bookings
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
				final_dataset.to_excel('data/Data_{}.xlsx'.format(c.replace(' ','')))
				return final_dataset
			else:
				isCompleted = 1
				print('No new data inserted')
				return dataset

	def stationarity(self):
		#different tests that check if the timeseries is stationary or not
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

	def MAPE(self, test, prediction):
		sum_ = 0
		for i in range(len(test)):
			diff = abs(test[i] - prediction[i])/abs(test[i])
			sum_ += diff

		MAPE = sum_ / len(test)
		return MAPE*100

	def model_fitting(self, train_size=24*14, test_size=24*7, cnt=0):
		ps = [1,2,3,4,5,6]
		ds = [0,1]
		qs = [1,2,3,4,5,6]
			
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

		for df in self.dataframes:
			#use 2 weeks for training and the 3rd week for testing
			df = df['Total'].to_numpy()
			train, test = df[0:train_size], df[train_size:train_size+test_size]
			for d in ds:
				for p in ps:
					for q in qs:
						print('ARIMA -> [{},{},{}]'.format(p,d,q))
						try:
							fc_ = []
							history = [x for x in train]
							for t in range(test_size):
								#forecast each new element of the testing set and use an expanding technique to update history
								model = ARIMA(history, order=(p,d,q))
								model_fit = model.fit(disp=-1)
								fc, se, conf = model_fit.forecast()
								fc_.append(fc[0])
								history.append(test[t])

							#calculate the MAPE and save it in a json file
							MAPE = self.MAPE(test, fc_)
							tmp_obj = {
								'P':p,
								'D':d,
								'Q':q,
								'MAPE':MAPE
							}
							obj[cnt][self.cities[cnt]].append(tmp_obj)
							print('Error: {}'.format(MAPE))
						except:
							#errors occur when the ARIMA model does not converge properly, so for some combination of parameters we cannot fit the model
							traceback.print_exc()
							tmp_obj = {
								'P':p,
								'D':d,
								'Q':q,
								'MAPE':'Impossible to fit'
							}
							obj[cnt][self.cities[cnt]].append(tmp_obj)

						with open('AllModelsAM.json','w') as f:
							f.write(json.dumps(obj, indent=4))

			cnt += 1

	def best_models(self, k=0.1):
		with open('AllModels.json') as f:
			models = json.loads(f.read())

		bestTO = {}
		bestAM = {}
		bestNY = {}

		#find the best parameters, looking also if the parameters are small enough
		for i in models:
			keyz = i.keys()
			best = 100
			if 'Torino' in keyz:
				for j in i['Torino']:
					try:
						if best - j['MAPE'] > k:
							best = j['MAPE']
							bestTO['P'] = j['P']
							bestTO['Q'] = j['Q']
							bestTO['D'] = j['D']
							bestTO['MAPE'] = j['MAPE']
					except:
						pass
			elif 'Amsterdam' in keyz:
				for j in i['Amsterdam']:
					try:
						if best -j['MAPE'] > k:
							best = j['MAPE']
							bestAM['P'] = j['P']
							bestAM['Q'] = j['Q']
							bestAM['D'] = j['D']
							bestAM['MAPE'] = j['MAPE']
					except:
						pass

			elif 'New York City' in keyz:
				for j in i['New York City']:
					try:
						if best -j['MAPE'] > k:
							best = j['MAPE']
							bestNY['P'] = j['P']
							bestNY['Q'] = j['Q']
							bestNY['D'] = j['D']
							bestNY['MAPE'] = j['MAPE']
					except:
						pass

		print(bestTO)
		print(bestAM)
		print(bestNY)

		return bestTO, bestAM, bestNY

	def prepare_models(self):
		bestTO, bestAM, bestNY = self.best_models()
		
		p_to, d_to, q_to = bestTO['P'], bestTO['D'], bestTO['Q']
		p_am, d_am, q_am = bestAM['P'], bestAM['D'], bestAM['Q']
		p_ny, d_ny, q_ny = bestNY['P'], bestNY['D'], bestNY['Q']
		MAPE_to, MAPE_am, MAPE_ny = bestTO['MAPE'], bestAM['MAPE'], bestNY['MAPE']
		ps = [p_to, p_am, p_ny]
		qs = [q_to, q_am, q_ny]
		ds = [d_to, d_am, d_ny]
		MAPEs = [MAPE_to, MAPE_am, MAPE_ny]

		return ps, ds, qs

	def expanding(self, train_start=24*7, test_size=24*7, n=24, cnt=0):
		ps, ds, qs = self.prepare_models()
		self.expanding_dict = {}
		for df in self.dataframes:
			df = df['Total'].to_numpy()
			train = df[0:train_start] #first 7 days of data
			self.expanding_dict[self.cities[cnt]] = []
			history = [x for x in train]
			for i in range(0,24*15,n):
				#for each training size, append data to history and save MAPE
				try:
					print(i)
					train = df[0:i+train_start]
					history = [x for x in train]
					fc_ = []
					for j in range(test_size):
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast()
						fc_.append(fc)
						history.append(df[i+train_start+j])

					MAPE = self.MAPE(df[i+train_start:i+train_start+test_size], fc_)
					self.expanding_dict[self.cities[cnt]].append(MAPE)
				except:
					#if model return an error, MAPE is equal to 100%
					MAPE = 100
					self.expanding_dict[self.cities[cnt]].append(MAPE)
					traceback.print_exc()
			cnt += 1

	def sliding(self, train_start=24*7, test_size=24*7, n=24, cnt=0):
		ps, ds, qs = self.prepare_models()
		sliding_size = range(0,24*15,n)
		self.sliding_dict = {}
		for df in self.dataframes:
			self.sliding_dict[self.cities[cnt]] = []
			df = df['Total'].to_numpy()
			for w in sliding_size:
				#for each training size, shift data and save MAPE
				try:
					print(w)
					train = df[0:w+train_start]
					history = [x for x in train]
					fc_ = []
					for i in range(test_size):
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast()
						fc_.append(fc[0])
						history.append(df[train_start+w+i])
						history = history[1:]
					test = df[w:w+test_size]
					MAPE = self.MAPE(test, fc_)
					self.sliding_dict[self.cities[cnt]].append(MAPE)
				except:
					MAPE = 100
					self.sliding_dict[self.cities[cnt]].append(MAPE)
					traceback.print_exc()
			cnt += 1

	def MAPE_best_strategy(self, test_size=24*7, cnt=0):
		ps,ds,qs = self.prepare_models()
		train_sizes = [24*13,24*17,24*14]
		for df in [self.NY]:
			df = df['Total'].to_numpy()
			train = df[0:train_sizes[cnt]]
			history = [x for x in train]
			fc_ = []
			for i in range(test_size):
				model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
				fc,conf,se = model.forecast()
				fc_.append(fc)
				history.append(df[train_sizes[cnt]+i])
				history = history[1:]

			test = df[train_sizes[cnt]:train_sizes[cnt]+test_size]
			MAPE = self.MAPE(test, fc_)
			print(MAPE)
			cnt += 1

	def horizon(self, test_size=24*7, cnt=0):
		#For the previous points, the best parameters for ps,ds,qs are found here
		ps, ds, qs = self.prepare_models()

		#Instead the best strategy is
		#Torino -> Sliding N=24*13
		#Amsterdam -> Sliding N=24*17
		#NewYorkCity -> Sliding N=24*18

		train_sizes = [24*13,24*17,24*18]
		ws = [i+1 for i in range(24)]
		for df in self.dataframes:
			df = df['Total'].to_numpy()
			errs = []
			for w in ws:
				print(w)
				train = df[0:train_sizes[cnt]]
				history = [x for x in train]
				fc_ = []
				for i in range(0,test_size,w):
					if i+w<168:
						# print(i)
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast(w)
						t = df[train_sizes[cnt]+i:train_sizes[cnt]+w+i]
						for f in range(len(fc)):
							fc_.append(fc[f])
							history.append(t[f])
							history = history[1:]
					else:
						#if the forecasted values are greater than 168 (test size), forecast only the missing ones
						j = 168-i
						model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
						fc,conf,se = model.forecast(j)
						t = df[train_sizes[cnt]+i:train_sizes[cnt]+j+i]
						for f in range(len(fc)):
							fc_.append(fc[f])
							history.append(t[f])
							history = history[1:]

				test = df[train_sizes[cnt]:train_sizes[cnt]+test_size]
				MAPE = self.MAPE(test, fc_)
				errs.append(MAPE)

			plt.figure(figsize=(15,6))
			plt.title('Horizon for {}'.format(self.cities[cnt].replace(" ","")))
			plt.ylabel('MAPE')
			plt.xlabel('Horizon Size')
			plt.plot(errs)
			plt.savefig('plots/Horizon{}.png'.format(self.cities[cnt].replace(" ","")))
			cnt += 1

	def plot_correlations(self, show=1, save=1, lags=48, cnt=0):
		for df in self.dataframes:
			data = df.loc[:, 'Total']
			fig, ax = plt.subplots(2)
			plot_acf(data, ax=ax[0], lags=lags, title='ACF for {}'.format(self.cities[cnt].replace(" ","")))
			plot_pacf(data, ax=ax[1], lags=lags, title='PACF')
			if show:
				plt.show()
			if save:
				plt.savefig('plots/ACF_PACF_{}.png'.format(self.cities[cnt].replace(" ","")))
			plt.close()
			cnt += 1

	def plot_timeseries(self, show=1, save=1, cnt=0):
		date = datetime.datetime(self.year, 1, 1)
		delta = int(self.dataframes[0].loc[0, 'Day']) - 1
		start = date + datetime.timedelta(days=delta)
		n = self.dataframes[0].loc[:, 'Day'].unique()
		x_lab = []
		for i in range(len(n)):
			x_lab.append(start.strftime('%d %b'))
			start = start + datetime.timedelta(days=1)
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

	def plot_error(self, show=1):
		with open('AllModels.json') as f:
			models = json.loads(f.read())

		to_models = models[0]['Torino']
		am_models = models[1]['Amsterdam']
		ny_models = models[2]['New York City']

		fig = plt.figure(figsize=(15,6))
		ax1 = plt.axes(frameon=True)

		y = []
		lbs = []
		xs = np.arange(0,72,1)
		for i in to_models:
			if i['MAPE'] != 'Impossible to fit':
				y.append(i['MAPE'])
				label = '[{},{},{}]'.format(i['P'], i['D'], i['Q'])
				lbs.append(label)
			else:
				y.append(np.nan)
				lbs.append('Model Error')
		mask = np.isfinite(np.array(y))
		x = np.arange(0,len(y), 1)
		plt.plot(xs[mask], np.array(y)[mask], '--o', label='Torino')
		for j in range(len(y)):
			if j%2==0:
				plt.annotate(lbs[j],
							(x[j], y[j]),
							textcoords="offset points", # how to position the text
                 			xytext=(0,5), # distance from text to points (x,y)
                 			ha='center')
		y = []
		lbs = []
		for i in am_models:
			if i['MAPE'] != 'Impossible to fit':
				y.append(i['MAPE'])
				label = '[{},{},{}]'.format(i['P'], i['D'], i['Q'])
				lbs.append(label)
			else:
				y.append(np.nan)
				lbs.append('Model Error')
		mask = np.isfinite(np.array(y))
		x = np.arange(0,len(y),1)
		plt.plot(xs[mask], np.array(y)[mask], '--o', color='red', label='Amsterdam')
		for j in range(len(y)):
			if j%2==0:
				plt.annotate(lbs[j],
							(x[j], y[j]),
							textcoords="offset points", # how to position the text
                 			xytext=(0,5), # distance from text to points (x,y)
                 			ha='center')

		y = []
		lbs = []
		for i in ny_models:
			if i['MAPE'] != 'Impossible to fit':
				y.append(i['MAPE'])
				label = '[{},{},{}]'.format(i['P'], i['D'], i['Q'])
				lbs.append(label)
			else:
				y.append(np.nan)
				lbs.append('Model Error')
		mask = np.isfinite(np.array(y))
		x = np.arange(0,len(y), 1)
		plt.plot(xs[mask], np.array(y)[mask], '--o', color='green', label='New York City', )
		for j in range(len(y)):
			if j%2==0:
				plt.annotate(lbs[j],
							(x[j], y[j]),
							textcoords="offset points", # how to position the text
                 			xytext=(0,5), # distance from text to points (x,y)
                 			ha='center')

		if show:
			# ax1.get_yaxis().tick_left()
			ax1.axes.get_xaxis().set_visible(False)
			plt.ylabel('MAPE [%]')
			plt.xlabel('Model Parameters')
			plt.legend()
			plt.savefig('plots/ErrorsModels.png')
			plt.show()

	def plot_heatmap(self):
		self.prepare_models()
		with open('AllModels.json') as f:
			obj = json.loads(f.read())

		for city in self.cities:
			df_to = np.zeros((7,7))
			to = obj[self.cities.index(city)][city]
			cnt = 36
			for i in range(df_to.shape[0]):
				for j in range(df_to.shape[1]):
					if i == 0:
						df_to[i,j] = None
					elif j == 0:
						df_to[i,j] = None

					elif to[cnt]['MAPE'] != 'Impossible to fit':
						df_to[i,j] = to[cnt]['MAPE']
						cnt += 1
					else:
						df_to[i,j] = None
						cnt += 1

			df_to = pd.DataFrame(df_to)
			plt.figure()
			ax = sb.heatmap(df_to, annot=True, fmt='.1f')
			plt.savefig('plots/Heatmap{}_D=1'.format(city.replace(' ','')))

	def plot_training_strategy(self):
		self.expanding()
		self.sliding()

		exp_dic = self.expanding_dict
		sli_dic = self.sliding_dict

		for i in self.cities:
			plt.figure(figsize=(15,6))
			plt.title('Training Strategy for {}'.format(i))
			plt.plot(exp_dic[i], label='Expanding Window')
			plt.plot(sli_dic[i], label='Sliding Window')
			plt.legend()
			# plt.xticks()
			plt.ylabel('MAPE')
			plt.savefig('plots/TrainingStrategy{}'.format(i.replace(" ","")))

	def plot_prediction(self, cnt=0):
		ps,ds,qs = self.prepare_models()
		cnt = 0
		for df in self.dataframes:
			res = ARIMA(df['Total'], order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
			fig, ax = plt.subplots(figsize=(15,6))
			ax = df.loc[680:, 'Total'].plot(ax=ax)
			# fig.figsize(15,6)
			fig = res.plot_predict(700,756, dynamic=False, ax=ax, plot_insample=False)
			plt.title('Prediction for {}'.format(self.cities[cnt]))
			plt.xlabel('Hours')
			plt.ylabel('Bookings')
			plt.savefig('plots/PlotPrediction{}'.format(self.cities[cnt].replace(' ','')))
			cnt += 1

	def plot_fitting(self, test_size=24*7, cnt=0):
		ps, ds, qs = self.prepare_models()
		train_sizes = [24*13,24*17,24*18]
		cnt = 0
		for df in self.dataframes:
			df = df['Total'].to_numpy()
			train = df[0:train_sizes[cnt]]
			history = [x for x in train]
			fc_ = []
			for i in range(test_size):
				print(round(i/test_size*100, 2))
				model = ARIMA(history, order=(ps[cnt], ds[cnt], qs[cnt])).fit(disp=-1)
				fc,se,conf = model.forecast()
				fc_.append(fc[0])
				history.append(df[train_sizes[cnt]+i])
				history = history[1:]

			#create a "bad" model to plot together with the best one
			train = df[0:train_sizes[cnt]]
			history = [x for x in train]
			fc_bad = []
			for i in range(test_size):
				print(round(i/test_size*100, 2))
				model = ARIMA(history, order=(1,0,1)).fit(disp=-1)
				fc,se,conf = model.forecast()
				fc_bad.append(fc)
				history.append(df[24*9+i])

			plt.figure(figsize=(15,6))
			plt.plot(df[train_sizes[cnt]:train_sizes[cnt]+24*7], label='Original', color='black')
			plt.plot(fc_bad, label='ARIMA Non-Optimal', linestyle='--')
			plt.plot(fc_, label='ARIMA [{},{},{}]'.format(ps[cnt],ds[cnt],qs[cnt], color='crimson', linestyle=''))
			plt.legend()
			plt.xlabel('Hours')
			plt.ylabel('Bookings')
			plt.title('Original vs Model for {}'.format(self.cities[cnt]))
			plt.savefig('plots/OriginalVsModel{}'.format(self.cities[cnt].replace(' ','')))
			cnt += 1