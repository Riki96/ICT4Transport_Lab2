import pymongo as pm
import datetime
from scipy import stats
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import random
import statsmodels as sm
import seaborn as sb
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA

class PreProcessing :
	def __init__(self):
		client = pm.MongoClient('bigdatadb.polito.it',
			ssl=True,
			authSource = 'carsharing',
			tlsAllowInvalidCertificates=True
			)
		self.db = client['carsharing']
		self.db.authenticate('ictts', 'Ictts16!')
		self.per_bk = self.db['PermanentBookings']
		self.act_bk = self.db['ActiveBookings']

	def mongoDB(self, city, unix_start, unix_end):
		pipeline = [{
					'$match':{
						'city':city,
						'init_time':{'$gte':unix_start,'$lte':unix_end}
						}
					},
						#FILTERING PORTION OF PIPELINE-----------------
					{
						'$project':{
						'init_date':1,
						'init_time':1,
						'final_time':1,
						'plate':1,
						'city':1,
						'durata': { '$divide': [ { '$subtract': ["$final_time", "$init_time"] }, 60 ] },
						'dist_lat':{'$abs':{'$subtract': [{'$arrayElemAt':[{'$arrayElemAt': [ "$origin_destination.coordinates", 0]}, 0]}, {'$arrayElemAt':[{'$arrayElemAt': [ "$origin_destination.coordinates", 1]}, 0]}]}},
						'dist_long':{'$abs':{'$subtract': [{'$arrayElemAt':[{'$arrayElemAt': [ "$origin_destination.coordinates", 0]}, 1]}, {'$arrayElemAt':[{'$arrayElemAt': [ "$origin_destination.coordinates", 1]}, 1]}]}},
						}
					},
					{
						'$match':{
							'$or':[
								{'dist_long':{'$gte':0.0003}},
								{'dist_lat':{'$gte':0.0003}},
								],
							'durata':{'$lte':180,'$gte':2},
						}
					},
						#END OF FILTERING
					{ 
						"$group": {
							"_id":{
								'hour':{'$hour':'$init_date'},
								'day':{'$dayOfYear':'$init_date'},
								# 'plate':'$plate'
								},
							's':{'$sum':1}	
    					}
		        	}]

		result = self.per_bk.aggregate(pipeline)
		output = list(result)
		# print(output)
		return output

	def dataset_creation(self, self.cities, start, end, startNY, endNY):
		for c in self.cities:
			print(c)
			if c == 'New York City':
				start = startNY
				end = endNY

			unix_start = time.mktime(start.timetuple())
			unix_end = time.mktime(end.timetuple())

			data = self.mongoDB(c, unix_start, unix_end)
			# exit()
			df = pd.DataFrame({
				'Day':[i['_id']['day'] for i in data],
				'Hour':[i['_id']['hour'] for i in data],
				'Total':[i['s'] for i in data]
				})
			df = df.sort_values(['Day', 'Hour'])
			print(df)
			c = c.strip()
			df.to_excel('data/Data_{}.xlsx'.format(c))

class DataMining ():
	def __init__(self, cities):
		sb.set_style('darkgrid')
		self.cities = cities
		self.time_series ={}
	def data_filling(self, dataset, c):
		hours = []
		for i in range(31):
			for j in range(24):
				hours.append(j)
		cnt = 0
		added = 0
		# print(len(hours))
		tmp_df = []
		for row, col in dataset.iterrows():
			# print(dataset.loc[row, 'Hour'])
			try:
				if dataset.loc[row+1, 'Hour'] - col['Hour'] > 1:
					for k in range(col['Hour']+1, dataset.loc[row+1, 'Hour']):
						tmp_row = [col['Day'], col['Hour'] + k,
									(col['Total']+k + dataset.loc[row+1, 'Total'])/2]
						print(tmp_row)
						tmp_df.append(tmp_row)	
			except:
				print('No Index')
		# print(tmp_df)
		tmp_df = pd.DataFrame(tmp_df, columns=['Day', 'Hour', 'Total'])
		final_dataset = pd.concat([dataset.loc[:, 'Day':'Total'], tmp_df], ignore_index=True)
		final_dataset = final_dataset.sort_values(['Day', 'Hour'])
		print(final_dataset)
		final_dataset.to_excel('data/Data_{}.xlsx'.format(c))
		return final_dataset
	
	def system_plotting(self,date):
		x_lab=[]
		for i in range(31):
			date = date + datetime.timedelta(days=i)
			x_lab.append(str('%r %d'%(date.strftime("%a"),i+1)))
			# x_lab.append(str('%r %d'%(date.strftime("%a"),i+1)))
		Torino, Amsterdam, New_York = self.files_opening()

		vect = [Torino, Amsterdam, New_York]
		cnt=0
		for i in vect:
			x_axis = [x*24 for x in range(31)]
			# x_lab = ['Day %d'%i for i in range(1,32)]
			print(len(i))
			plt.figure(figsize=(15,5))
			plt.plot(list(i.loc[:,'Total']))
			plt.title(self.cities[cnt]+' time series')
			plt.grid()
			plt.xticks(x_axis,x_lab, rotation=45)
			plt.show()
			cnt+=1

	def plot_correlations(self, plot=True, save=True):
		df = pd.read_excel('data/Data_Torino.xlsx')
		data = df.loc[:, 'Total']

		fig, ax = plt.subplots(2)
		ax[0] = plot_acf(data, ax=ax[0], lags=48, title='ACF')
		ax[1] = plot_pacf(data, ax=ax[1], lags=48, title='PACF')
		if plot:
			plt.show()
		if save:
			plt.savefig('plots/ACF_PACF.png')

	def files_opening(self):
		for filename in os.listdir('data'):
			if 'Torino' in filename:
				# print('Torino')
				Torino = pd.read_excel('data/'+filename)
				Torino = self.data_filling(Torino, 'Torino')
			elif 'Amsterdam' in filename:
				# print('Amsterdam')
				Amsterdam = pd.read_excel('data/'+filename)
				Amsterdam = self.data_filling(Amsterdam, 'Amsterdam')
			elif 'New York' in filename:
				# print('New York')
				NY = pd.read_excel('data/'+filename)
				NY = self.data_filling(NY, 'New York City')
		return Torino, Amsterdam, NY
	

	def arima_training(self):
		df_TO, df_AM, df_NY = self.files_opening()
		dataframe = [df_TO, df_AM, df_NY]
		
		lag_order = (1,2,3,4)
		MA_order = (1,2,3,4)
		#lag_order = (1,2)
		#MA_order = (1,2)
		results = {}


		
		cnt = 0
		for df in dataframe:
			df = df['Total']
			df += np.random.random_sample(len(df))/10
			X = df.values.astype(float)
			self.time_series[self.cities[cnt]]= X
			train_size = 24*7*3
			test_len = len(df.index) - train_size
			predictions = np.zeros((len(lag_order),len(MA_order),test_len)) # to store outputs
			#df.head()
			#print (X)
			print('****************')
			print ('Study case:', self.cities[cnt])
			print('****************')


			train, test = X[0:train_size], X[train_size:(train_size+test_len)]
			history = [x for x in train]

			for p in lag_order:
				for q in MA_order:
					print('Testing ARIMA order (%i,%i,%i)' % (p,0,q))

					for t in range(0, test_len):
					
						model = ARIMA(history,order=(p,0,q))
						model_fit = model.fit(disp=0, maxiter=500, method='css')
						output = model_fit.forecast()
						yhat = output[0] 
						predictions[lag_order.index(p)][MA_order.index(q)][t]=yhat
						obs = test[t] # slide over time, by putting now+1 into past
						#print('--> t=%i: prediction=%.2f, actual=%i, error=%1.2f (%.2f%%) -- ' % (t, yhat, obs, yhat-obs, 100*(yhat-obs)/obs))
						
						history.append(obs)
						history = history[1:] # linea sliding/ expanding, if expanding comment
			
			results[self.cities[cnt]] = self.performances(predictions,test,self.cities[cnt], lag_order, MA_order)
			
			cnt+=1

		print (results)
		return results
			

	def performances(self,predictions,test,city, lag_order,MA_order):
		order_list =[]
		MAE_list =[]
		MAPE_list =[]
		MSE_list =[]
		R2_list =[]

		for p in lag_order: # now for each model compute performance and plot predictions
			plt.figure(figsize=(15,5))
			plt.plot(test, color='black', label='Orig', linewidth = 2)# plot the real time series
			
			for q in MA_order:
				order_list.append('('+str(p)+',0,'+str(q)+')')
				MAE_list.append(mean_absolute_error(test, predictions[lag_order.index(p)][MA_order.index(q)]))
				MAPE_list.append(mean_absolute_error(test, predictions[lag_order.index(p)][MA_order.index(q)]) / test.mean()*100)
				MSE_list.append(mean_squared_error(test, predictions[lag_order.index(p)][MA_order.index(q)]))
				R2_list.append(r2_score(test, predictions[lag_order.index(p)][MA_order.index(q)]))

				plt.plot(predictions[lag_order.index(p)][MA_order.index(q)], label='p=%i,q=%i' % (p,q) ) 
			
			plt.legend()
			plt.title(str(city)+': PREDICTION with p='+str(p))
			plt.savefig('plots/model_fitting_'+str(city)+'_P='+str(p)+'.png')

		performances = pd.DataFrame({	
										'order': order_list,
										'MAE': MAE_list,
										'MAPE': MAPE_list ,
										'MSE': MSE_list,
										'R2': R2_list
									})
		#print (performances)

		return performances.loc[performances['MSE'].idxmin()]['order']

		
		#return city, P, Q


	def results_predicting(self):
		orders = self.arima_training()

		for c in self.cities:
			model = ARIMA(self.time_series[c], order=orders[c])
			fig, ax = plt.subplots()
			ax = df.loc[650:].plot(ax=ax) # plot the data from step 650 up to now
			#	plot now the predicted data for the future days – we have 720 sample – so let’s go up to 780 
			fig = model_fit.plot_predict(650, 780, dynamic=False, ax=ax, plot_insample=False)











