#!/usr/bin/python3.4
#-*-coding:UTF-8-*-

import numpy as np
import csv
import sys

# read data from file.
capital_max_g = 7000
capital_max_l = 390
trX_row = 22792 # 32561
trX_v_row = 9769
teX_row = 16281
t_col = 115
rawdata_trX = np.zeros((trX_row,t_col))
rawdata_trX_v = np.zeros((trX_v_row,t_col))
rawdata_trY = np.zeros((trX_row,1))
rawdata_trY_v = np.zeros((trX_v_row,1))
rawdata_teX = np.zeros((teX_row,t_col))

filename1 = sys.argv[3] # 'X_train'
filename2 = sys.argv[4] # 'Y_train'
filename3 = sys.argv[5] # 'X_test'
filename4 = sys.argv[6] # 'SUB_15'

file1 = open(filename1,'r+',encoding='Big5')
reader1 = csv.reader(file1)
for row in reader1:
	if(reader1.line_num==1):
		pass
	elif(reader1.line_num<=trX_row+1):
		rawdata_trX[reader1.line_num-2,:101] = [row[2]]+row[6:]
		rawdata_trX[reader1.line_num-2,101] = 1 if int(row[0])<=25 else 0
		rawdata_trX[reader1.line_num-2,102] = 1 if (int(row[0])>=26 and int(row[0])<=45) else 0
		rawdata_trX[reader1.line_num-2,103] = 1 if (int(row[0])>=46 and int(row[0])<=65) else 0
		rawdata_trX[reader1.line_num-2,104] = 1 if (int(row[0])>=66) else 0
		rawdata_trX[reader1.line_num-2,105] = 1 if int(row[5])<=25 else 0
		rawdata_trX[reader1.line_num-2,106] = 1 if (int(row[5])>=26 and int(row[5])<=40) else 0
		rawdata_trX[reader1.line_num-2,107] = 1 if (int(row[5])>=41 and int(row[5])<=60) else 0
		rawdata_trX[reader1.line_num-2,108] = 1 if (int(row[5])>=61) else 0
		rawdata_trX[reader1.line_num-2,109] = 1 if int(row[3])==0 else 0
		rawdata_trX[reader1.line_num-2,110] = 1 if (int(row[3])>0 and int(row[3])<capital_max_g) else 0
		rawdata_trX[reader1.line_num-2,111] = 1 if int(row[3])>=capital_max_g else 0
		rawdata_trX[reader1.line_num-2,112] = 1 if int(row[4])==0 else 0
		rawdata_trX[reader1.line_num-2,113] = 1 if (int(row[4])>0 and int(row[4])<capital_max_l) else 0
		rawdata_trX[reader1.line_num-2,114] = 1 if int(row[4])>=capital_max_l else 0
	else:
		rawdata_trX_v[reader1.line_num-(trX_row+2),:101] = [row[2]]+row[6:]
		rawdata_trX_v[reader1.line_num-(trX_row+2),101] = 1 if int(row[0])<=25 else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),102] = 1 if (int(row[0])>=26 and int(row[0])<=45) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),103] = 1 if (int(row[0])>=46 and int(row[0])<=65) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),104] = 1 if (int(row[0])>=66) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),105] = 1 if int(row[5])<=25 else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),106] = 1 if (int(row[5])>=26 and int(row[5])<=40) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),107] = 1 if (int(row[5])>=41 and int(row[5])<=60) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),108] = 1 if (int(row[5])>=61) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),109] = 1 if int(row[3])==0 else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),110] = 1 if (int(row[3])>0 and int(row[3])<capital_max_g) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),111] = 1 if int(row[3])>=capital_max_g else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),112] = 1 if int(row[4])==0 else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),113] = 1 if (int(row[4])>0 and int(row[4])<capital_max_l) else 0
		rawdata_trX_v[reader1.line_num-(trX_row+2),114] = 1 if int(row[4])>=capital_max_l else 0
file1.close()
file2 = open(filename2,'r+',encoding='Big5')
reader2 = csv.reader(file2)
for row in reader2:
	if(reader2.line_num<=trX_row):
		rawdata_trY[reader2.line_num-1,:] = row
	else:
		rawdata_trY_v[reader2.line_num-(trX_row+1),:] = row
file2.close()
file3 = open(filename3,'r+',encoding='Big5')
reader3 = csv.reader(file3)
for row in reader3:
	if(reader3.line_num==1):
		pass
	else:
		rawdata_teX[reader3.line_num-2,:101] = [row[2]]+row[6:]
		rawdata_teX[reader3.line_num-2,101] = 1 if int(row[0])<=25 else 0
		rawdata_teX[reader3.line_num-2,102] = 1 if (int(row[0])>=26 and int(row[0])<=45) else 0
		rawdata_teX[reader3.line_num-2,103] = 1 if (int(row[0])>=46 and int(row[0])<=65) else 0
		rawdata_teX[reader3.line_num-2,104] = 1 if (int(row[0])>=66) else 0
		rawdata_teX[reader3.line_num-2,105] = 1 if int(row[5])<=25 else 0
		rawdata_teX[reader3.line_num-2,106] = 1 if (int(row[5])>=26 and int(row[5])<=40) else 0
		rawdata_teX[reader3.line_num-2,107] = 1 if (int(row[5])>=41 and int(row[5])<=60) else 0
		rawdata_teX[reader3.line_num-2,108] = 1 if (int(row[5])>=61) else 0
		rawdata_teX[reader3.line_num-2,109] = 1 if int(row[3])==0 else 0
		rawdata_teX[reader3.line_num-2,110] = 1 if (int(row[3])>0 and int(row[3])<capital_max_g) else 0
		rawdata_teX[reader3.line_num-2,111] = 1 if int(row[3])>=capital_max_g else 0
		rawdata_teX[reader3.line_num-2,112] = 1 if int(row[4])==0 else 0
		rawdata_teX[reader3.line_num-2,113] = 1 if (int(row[4])>0 and int(row[4])<capital_max_l) else 0
		rawdata_teX[reader3.line_num-2,114] = 1 if int(row[4])>=capital_max_l else 0
file3.close()

# feature normalization
rawdata_trX_sum = np.sum(rawdata_trX,0)
rawdata_trX_mean = rawdata_trX_sum/trX_row
rawdata_trX_std = np.std(rawdata_trX,0)
rawdata_trX = (rawdata_trX-rawdata_trX_mean)/rawdata_trX_std

rawdata_trX_v_sum = np.sum(rawdata_trX_v,0)
rawdata_trX_v_mean = rawdata_trX_v_sum/trX_v_row
rawdata_trX_v_std = np.std(rawdata_trX_v,0)
rawdata_trX_v_std = np.array([1 if i==0 else i for i in rawdata_trX_v_std])
rawdata_trX_v = (rawdata_trX_v-rawdata_trX_v_mean)/rawdata_trX_v_std

rawdata_teX_sum = np.sum(rawdata_teX,0)
rawdata_teX_mean = rawdata_teX_sum/teX_row
rawdata_teX_std = np.std(rawdata_teX,0)
rawdata_teX_std = np.array([1 if i==0 else i for i in rawdata_teX_std])
rawdata_teX = (rawdata_teX-rawdata_teX_mean)/rawdata_teX_std

#  Losgistic Regression
weight_d = np.zeros((t_col,1))
bias_d = 0.0
par_weight_d = np.zeros((t_col,1))
par_bias_d = 0.0
hist_par_weight_d = np.zeros((t_col,1))
hist_par_bias_d = 0.0

epoch = 500
batch_size = 3100
lrate_d = 0.0001
best_cor = 0.0

shuffle_d = np.arange(int(trX_row/batch_size))
np.random.seed(1064)
for it in range(epoch):
	np.random.shuffle(shuffle_d)
	for itn in range(int(trX_row/batch_size)):
		diff_y_fx =rawdata_trY[shuffle_d[itn]*batch_size:(shuffle_d[itn]+1)*batch_size,:]-(1/(1+np.exp(-1*(rawdata_trX[shuffle_d[itn]*batch_size:(shuffle_d[itn]+1)*batch_size,:].dot(weight_d)+bias_d*np.ones((batch_size,1))))))
		par_weight_d = -1*lrate_d*rawdata_trX[shuffle_d[itn]*batch_size:(shuffle_d[itn]+1)*batch_size,:].T.dot(diff_y_fx)
		par_bias_d = -1*lrate_d*np.sum(diff_y_fx)
		hist_par_weight_d = np.sqrt(hist_par_weight_d**2+par_weight_d**2)
		hist_par_bias_d = np.sqrt(hist_par_bias_d**2+par_bias_d**2)
		weight_d -= par_weight_d/hist_par_weight_d
		bias_d -= par_bias_d/hist_par_bias_d
	if(it%1==0):
		print_ans = 1/(1+np.exp(-1*(rawdata_trX_v.dot(weight_d)+bias_d*np.ones((trX_v_row,1)))))
		print_ans = [1 if i >= 0.5 else 0 for i in print_ans]
		cor_ans = [0 if print_ans[i]!=rawdata_trY_v[i] else 1 for i in range(len(print_ans))]
		cor = np.sum(cor_ans)
		# print("At "+str(it)+" epoch, the overall Correctness is "+str(cor/trX_v_row)+".")
		if(cor/trX_v_row > best_cor):
			best_cor = cor/trX_v_row
			best_weight_d = weight_d
			best_bias_d = bias_d

# print("The best score is "+str(best_cor))
weight_d = best_weight_d
bias_d = best_bias_d

# Prediction for test set
ansdata_teY = 1/(1+np.exp(-1*(rawdata_teX.dot(weight_d)+bias_d*np.ones((teX_row,1)))))
ansdata_teY = [1 if i >= 0.5 else 0 for i in ansdata_teY]
file4 = open(filename4,'w+',encoding='Big5')
writer4 = csv.writer(file4)
writer4.writerow(['id','label'])
for it in range(teX_row):
	writer4.writerow([str(it+1),ansdata_teY[it]])
file4.close()


