import numpy as np
import csv
import sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
#categorical_crossentropy

tst_img_num = 7178
img_rows = 48
img_cols = 48
filename2 = sys.argv[1] # "test.csv"
filename3 = sys.argv[2] # "SUB0.csv"
modelname = 'model_best.h5'

def load_data():
	x_test = np.zeros((tst_img_num, img_rows*img_cols))
	
	file2 = open(filename2,'r',encoding='UTF-8')
	reader2 = csv.reader(file2)
	next(reader2)		# Now, line_num starts from two.
	for row in reader2:
		x_test[reader2.line_num-2,] = np.array([int(i) for i in row[1].split()])

	x_test = x_test.astype('float32')
	x_test = x_test/255
	
	return x_test
x_test= load_data()

x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)


model2 = load_model(modelname)
ans = model2.predict(x_test, batch_size=125)
# ans[0]= [  8.00232738e-02   7.85117509e-20   1.46075207e-10   3.02247843e-03	9.16945279e-01   2.68015459e-08   9.00301711e-06]
ans = np.array([idx for arr in ans for idx,val in enumerate(arr) if val== max(arr)]).reshape(tst_img_num,1)

# Write prediction to file.
file3 = open(filename3,'w+',encoding='UTF-8')
writer3 = csv.writer(file3)
writer3.writerow(["id","label"])
for i in range(len(ans)):
	writer3.writerow([i,ans[i][0]])
file3.close()



