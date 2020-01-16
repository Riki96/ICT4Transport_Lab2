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
		re = (abs(test.subtract(prediction))/abs(test)).sum()
		mre = re/test.mean()
		return mre

	def model_fitting(self, train_size=24*14, append=0, sliding=0, start=650):
		ps = [1,2,3,4,5,6,7]
		ds = [0,1]
		qs = [1,2,3,4,5,6,7]

		# with open('AllModels_2.json') as f:
			# obj = json.loads(f.read()) #file in which all the models performances will be saved
			
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
		# self.best_order = {} #object to save only the model with the lowest MRE
		len_test = 24*7
		for df in self.dataframes:
			df = df.loc[:, 'Total']
			train, test = df[0:train_size], df[train_size:train_size+len_test]
			mre_min = 1e5
			history = [x for x in train]
			for d in ds:
				for p in ps:
					for q in qs:
						yhats = []
						print('ARIMA -> [{},{},{}]'.format(p,d,q))
						try:
							model = ARIMA(history, order=(p,d,q))
							model_fit = model.fit(disp=0)
							fc, se, conf = model_fit.forecast(len(test.index))
							fc_s = pd.Series(fc, index=test.index)
							lowers = pd.Series(conf[:,0], index=test.index)
							uppers = pd.Series(conf[:,1], index=test.index)
							mre = self.mean_relative_error(test, fc_s)
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

	def error_plotting(self):
		with open('AllModels_2.json') as f:
			models = json.loads(f.read())

		to_models = models['Torino']
		am_models = models['Amsterdam']
		ny_models = models['New York City']

		plt.figure()
		xtick = []
		y = []
		for i in to_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])
				# xtick.append('p={}-d={}-q={}'.format(i['P'],i['D'],i['Q']))
		# y = [i for i in y if i<100]
		# x = [i for i in range(len(y))]
		# plt.xticks(x, xtick)
		plt.plot(y)

		xtick = []
		y = []
		for i in am_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])
				# xtick.append('p={}-d={}-q={}'.format(i['P'],i['D'],i['Q']))
		# y = [i for i in y if i<100]
		# x = [i for i in range(len(y))]
		# plt.xticks(x, xtick)
		plt.plot(y, color='red')

		xtick = []
		y = []
		for i in ny_models:
			if i['MRE'] != 'Impossible to fit':
				y.append(i['MRE'])
				# xtick.append('p={}-d={}-q={}'.format(i['P'],i['D'],i['Q']))
		# y = [i for i in y if i<100]
		# x = [i for i in range(len(y))]
		# plt.xticks(x, xtick)
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
			best = 1e5
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

	
	def expanding(self, last_training_day=21):
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
		for c in range(2,3):
			df = self.dataframes[c]
			train = df[0:24*7] #first 7 days of data
			test = df[24*last_training_day::]
			errs = []
			index = 24*7+1
			for i in range(8,last_training_day+1):
				for j in range(24):
					history = [x for x in train['Total']]
					try:	
						model = ARIMA(history, order=(ps[c],ds[c],qs[c])).fit(disp=0)
						# fc,_,_ = model.forecast(len(test.index))
						# fc_s = pd.Series(fc, index=test.index)
						fc,_,_ = model.forecast()
						fc_s = pd.Series(fc, index=test.index)
						mre = self.mean_relative_error(test['Total'], fc_s)
						print(mre)
						errs.append(mre)
					except:
						traceback.print_exc()

					# new_values = pd.DataFrame([df.lo])
					train = train.append(df[(i-1)*24:i*24], ignore_index=True)
				# print(train)
				# exit()

			print(mres[c])
			plt.figure()
			plt.title('Sliding Strategy [W=24 hours]')
			plt.ylabel('Mean Relative Error')
			plt.plot(errs)
			plt.savefig('plots/ExpandingStrategy{}'.format(self.cities[c]))


	def sliding(self, last_training_day=21):
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

		for c in range(3):
			df = self.dataframes[c]
			train = df.loc[0:7*24, 'Total']
			test = df.loc[last_training_day*24::]
			errs = []
			for i in range(8*24, (last_training_day+1)*24):
				history = [x for x in train]
				try:	
					model = ARIMA(history, order=(ps[c],ds[c],qs[c])).fit(disp=0)
					fc,_,_ = model.forecast(len(test.index))
					fc_s = pd.Series(fc, index=test.index)
					mre = self.mean_relative_error(test['Total'], fc_s)
					print(mre)
					errs.append(mre)
				except:
					# traceback.print_exc()
					pass

				new_values = df.loc[i:i+1, 'Total']
				train = train.append(new_values)
				train = train.drop(train.index[0])
			print(mres[c])
			plt.figure()
			plt.title('Expanding Strategy [W=24 hours]')
			plt.ylabel('Mean Relative Error')
			plt.plot(errs)
			plt.savefig('plots/SlidingStrategy{}'.format(self.cities[c].strip()))





