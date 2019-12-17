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
		
		self.act_bk = self.db['ActiveBookings']


		init_date =datetime.datetime(2017,10,1)
		final_date =  datetime.datetime(2017,10,31)

		

	def mongoDB():
		pipeline = [{
					'$match': {
						'city': city,
						''
					},

		}]


	def cleaning():

	def creating_dataset():



class DataMining ():
	def __init__(self,dataset):
		self.data = dataset

	def model_specifications():

	def performances():

	def prediction_results():
