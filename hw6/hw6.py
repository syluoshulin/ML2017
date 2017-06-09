import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Concatenate, Dot, Merge, Add
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from movies_class import data_process

EPOCH = 200
BATCH_SIZE = 256
VALID_SPLIT = 0.1

def get_model(n_users, n_items, latent_dim=100):
	classMovies = data_process()
	inputUser = Input(shape=[1])
	inputItem = Input(shape=[1])

	userVec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(inputUser)
	userVec = Flatten()(userVec)

	itemVec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(inputItem)
	itemVec = Flatten()(itemVec)

	userBias = Embedding(n_users, 1, embeddings_initializer='zeros')(inputUser)
	userBias = Flatten()(userBias)

	itemBias = Embedding(n_users, 1, embeddings_initializer='zeros')(inputItem)
	itemBias = Flatten()(itemBias)

	rHat = Dot(axes=1)([userVec, itemVec])
	rHat = Add()([rHat, userBias, itemBias])

	model = Model([inputUser, inputItem], rHat)
	
	model.compile(loss='mse', optimizer='sgd',metrics=[classMovies.root_mean_squared_error])
	model.summary()

	return model	

def nn_model(n_users, n_items, latent_dim=100):
	classMovies = data_process()
	inputUser = Input(shape=[1]) 
	inputItem = Input(shape=[1])

	userVec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(inputUser)
	userVec = Flatten()(userVec)

	itemVec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(inputItem)
	itemVec = Flatten()(itemVec)

	mergeVec = Concatenate()([userVec, itemVec])
	hidden = Dense(150,Activation='relu')(mergeVec)
	hidden = Dense(50,Activation='relu')(hidden)
	output = Dense(1)(hidden)

	model = keras.models.Model([inputUser, inputItem], output)

	model.compile(loss='mse', optimizer='sgd',metrics=[classMovies.root_mean_squared_error])
	model.summary()

	return model

if __name__ == "__main__":
	classMovies = data_process()
	dirBase, dirExp = classMovies.get_path()
	path, history_data = classMovies.history_data(dirExp, EPOCH)

	#dataRating = pd.read_csv(os.path.join(sys.argv[1],'train.csv'), sep=',', engine='python')
	#dataUsers = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
	#dataMovies = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

	dataTest = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')
	testUsers = dataTest.UserID.astype('category')
	testMovies = dataTest.MovieID.astype('category')

	#print data.head()
	#data.set_index(data.columns[0], inplace=True) #replace index with the first row
	#print data[data.columns[0]].reshape(len(data[data.columns[0]]),1)

	#print("Data loading...")
	#print('Training data: ', dataRating.shape)
	#print('Users data: ', dataUsers.shape)
	#print('Movies data: ', dataMovies.shape)

	#dataMovies['Genres'] = dataMovies.Genres.str.split('|')
	#dataUsers.Age = dataUsers.Age.astype('category')
	#dataUsers.Gender = dataUsers.Gender.astype('category')
	#dataUsers.Occupation = dataUsers.Occupation.astype('category')
	#dataRating.MovieID = dataRating.MovieID.astype('category')
	#dataRating.UserID = dataRating.UserID.astype('category')

	numMovies = 3952
	numUsers = 6040
	#movieID = dataRating.MovieID.values
	#userID = dataRating.UserID.values

	#yData = np.zeros((dataRating.shape[0], 5))
	#Rating need to minus by 1, due to set range from 1~5 to 0~4
	#yData[np.arange(dataRating.shape[0]), dataRating.Rating - 1] = 1

	#Normalize
	#yData = dataRating.Rating.values
	#yMean = np.mean(yData)
	#yStd = np.std(yData)
	#yData = (yData-yMean)/yStd

	#xTrain1, xTrain2, yTrain, xVal1, xVal2, yVal = classMovies.split_data(userID, movieID, yData, VALID_SPLIT)

	model = get_model(numUsers, numMovies,latent_dim=64)

	#checkpoint = ModelCheckpoint(filepath='./hw6_mf_reg_best.h5', verbose=1,\
	#							save_best_only=True,\
	#							monitor='val_root_mean_squared_error',\
	#							mode='min')

	#csv_logger = CSVLogger(os.path.join(path,'record_reg.log'))
	#model.fit([xTrain1, xTrain2], yTrain, validation_data=([xVal1, xVal2], yVal),\
											#batch_size=BATCH_SIZE,\
											#epochs=EPOCH,\
											#callbacks=[checkpoint,csv_logger])


	#model.save('./hw6_mf_reg.h5')
	#plot_model(model,to_file='./hw6_mf_reg.png')

	model = load_model('./hw6_model.h5',\
						custom_objects={'root_mean_squared_error': classMovies.root_mean_squared_error})
	yTest = model.predict([testUsers, testMovies])
	yMean = 3.58171208604 
	yStd = 1.11689766115 
	yTest = yTest * yStd + yMean
	# yTest = np.argmax(yTest, 1) + 1
	
	outputTest = dataTest
	outputTest['UserID'] = yTest
	outputTest = outputTest.drop('MovieID', 1)
	outputTest.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
	

