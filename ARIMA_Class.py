import pymongo as pm
import datetime
from scipy import stats
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sb
import os

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
		return output

	def dataset_creation(self, cities, start, end, startNY, endNY):
		for c in cities:
			print(c)
			if c == 'New York City':
				start = startNY
				end = endNY

			unix_start = time.mktime(start.timetuple())
			unix_end = time.mktime(end.timetuple())

			data = self.mongoDB(c, unix_start, unix_end)
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
	def __init__(self):
		self.d = d
		self.p = p
		self.q = q
	
	def system_plotting(self,cities,date):
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
			plt.title(cities[cnt]+' time series')
			plt.grid()
			plt.xticks(x_axis,x_lab, rotation=45)
			plt.show()
			cnt+=1

	def acf(self):
		df = pd.read_excel('data/Data_Torino.xlsx')
		data = df.loc[:, 'Total']
		ACF = acf(data, nlags=736)
		PACF = pacf(data, nlags=736)
		plt.figure()
		plt.plot(PACF)

	def files_opening(self):
		for filename in os.listdir('data'):
			if 'Torino' in filename:
				print ('Torino')
				Torino = pd.read_excel('data/'+filename)
			elif 'Amsterdam' in filename:
				print ('Amsterdam')
				Amsterdam = pd.read_excel('data/'+filename)
			elif 'New York' in filename:
				print ('New York')
				NY = pd.read_excel('data/'+filename)


		return Torino, Amsterdam, NY
	

	def model_specifications():
		pass

	def performances():
		pass

	def prediction_results():
		pass
