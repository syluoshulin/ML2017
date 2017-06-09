import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Concatenate, Dot, Merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from movies_class import data_process

epochs = 100
batch_size = 256
validation_split = 0.1

if __name__ == "__main__":
    movies_class = data_process()
    base_dir, exp_dir = movies_class.get_path()
    store_path, history_data = movies_class.history_data(exp_dir, epochs)

    # ratings_data = pd.read_csv(os.path.join(sys.argv[1],'train.csv'), sep=',', engine='python')
    #users_data = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    #movies_data = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

    test_data = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')
    test_users = test_data.UserID.astype('category')
    test_movies = test_data.MovieID.astype('category')

    #print data.head()
    #data.set_index(data.columns[0], inplace=True) #replace index with the first row
    #print data[data.columns[0]].reshape(len(data[data.columns[0]]),1)

    #print("Data loading...")
    #print('Training data: ', ratings_data.shape)
    #print('Users data: ', users_data.shape)
    #print('Movies data: ', movies_data.shape)

    #movies_data['Genres'] = movies_data.Genres.str.split('|')
    #users_data.Age = users_data.Age.astype('category')
    #users_data.Gender = users_data.Gender.astype('category')
    #users_data.Occupation = users_data.Occupation.astype('category')
    #ratings_data.MovieID = ratings_data.MovieID.astype('category')
    #ratings_data.UserID = ratings_data.UserID.astype('category')

    n_movies = 3952 #ratings_data['MovieID'].drop_duplicates().max()#movies_data.shape[0]
    n_users = 6040 #ratings_data['UserID'].drop_duplicates().max()#users_data.shape[0]
    #movieID = ratings_data.MovieID.values
    #userID = ratings_data.UserID.values

    #Y_data = np.zeros((ratings_data.shape[0], 5))
    #Rating need to minus by 1, due to set range from 1~5 to 0~4
    #Y_data[np.arange(ratings_data.shape[0]), ratings_data.Rating - 1] = 1
    #Y_data = ratings_data.Rating.values

    #X1_train, X2_train, Y_train, X1_val, X2_val, Y_val = movies_class.split_data(userID, movieID, Y_data, validation_split)

    #MODEL
    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users+1, 256, input_length=1)(user_input)
    user_vec = Flatten()(user_embedding)
    user_vec = Dropout(0.2)(user_vec)

    movie_input = Input(shape=[1])
    movie_embedding = Embedding(n_movies+1, 256, input_length=1)(movie_input)
    movie_vec = Flatten()(movie_embedding)
    movie_vec = Dropout(0.2)(movie_vec)

    vec_inputs = Concatenate()([user_vec, movie_vec])
    model = Dense(1024, activation='relu')(vec_inputs)
    model = Dropout(0.4)(model)
    #model = BatchNormalization()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.4)(model)
    #model = BatchNormalization()(model)
    model = Dense(32, activation='relu')(model)
    model = Dropout(0.4)(model)
    model_out = Dense(1, activation='linear')(model)

    
    model = Model([user_input, movie_input], model_out)
    model.compile(loss='mse', optimizer='rmsprop', metrics=[movies_class.root_mean_squared_error])

    #checkpoint = ModelCheckpoint(filepath='best_model.h5', verbose=1, save_best_only=True, monitor='val_root_mean_squared_error', mode='min')
    #model.fit([X1_train, X2_train], Y_train, validation_data=([X1_val, X2_val], Y_val), batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
    #model.save('model.h5')
    #plot_model(model,to_file='model.png')

    model = load_model('./hw6_best_model.h5', custom_objects={'root_mean_squared_error': movies_class.root_mean_squared_error})
    Y_test = model.predict([test_users, test_movies])
    
    #Y_test = np.argmax(Y_test, 1) + 1
    
    test_output = test_data
    test_output['UserID'] = Y_test
    test_output = test_output.drop('MovieID', 1)
    test_output.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
    
