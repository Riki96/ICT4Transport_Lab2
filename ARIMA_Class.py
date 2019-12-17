import pymongo as pm
import datetime
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

class PreProcessing :
	def __init__():
		
		client = pm.MongoClient('bigdatadb.polito.it',
			ssl=True,
			authSource = 'carsharing',
			tlsAllowInvalidCertificates=True
			)
		self.db = client['carsharing']
		self.db.authenticate('ictts', 'Ictts16!')

		self.per_bk = self.db['PermanentBookings']
		
		# self.act_bk = self.db['ActiveBookings']


		

	def mongoDB(self,city, start, end):
		pipeline = [{
						'$match':{
							'city':c,
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
    									'plate':'$plate'
    									}
    							
    						}
		        	}]

		result = self.per_bk.aggregate(pipeline)

	def dataset_creation(cities, start, end):
		for c in cities:
			if c == 'New York City':
				start = datetime.datetime(start, timedelta=-8)
				end = datetime.datetime(end, timedelta=-8)
			else:
				start = datetime.datetime(start)
				end = datetime.datetime(end)
			data = self.mongoDB(c, start, end)
			df = pd.DataFrame(data)
			c = c.trim()
			df.to_excel('Data_{}'.format(c))

class DataMining ():
	def __init__(self,dataset):
		self.data = dataset

	def model_specifications():

	def performances():

	def prediction_results():
