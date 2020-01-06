import pymongo as pm
import datetime
import time
import pandas as pd


class PreProcessing:
	def __init__(self, start, end, startNY, endNY):
		self.cities = ['Torino', 'Amsterdam', 'New York City']
		self.start = start
		self.end = end
		self.startNY = startNY
		self.endNY = endNY

	def run(self):
		client = pm.MongoClient('bigdatadb.polito.it',
			ssl=True,
			authSource = 'carsharing',
			tlsAllowInvalidCertificates=True
			)
		self.db = client['carsharing']
		self.db.authenticate('ictts', 'Ictts16!')
		print('Autentication Completed!')
		self.per_bk = self.db['PermanentBookings']

	def mongoDB(self, city, unix_start, unix_end):
		print('Request {} started'.format(city))
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
							'duration': { '$divide': [ { '$subtract': ["$final_time", "$init_time"] }, 60 ] },
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
							'duration':{'$lte':180,'$gte':2},
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
		print('Request {} ended'.format(city))
		output = list(result)
		# print(output)
		return output

	def dataset_creation(self):
		self.run()
		for c in self.cities:
			print(c)
			if c == 'New York City':
				start = self.startNY
				end = self.endNY
			else:
				start = self.start
				end = self.end

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
			print('DataFrame Created and Saved')
			c = c.strip()
			df.to_excel('data/Data_{}.xlsx'.format(c))