#!/usr/bin/python3.4
#-*-coding:UTF-8-*-

import numpy as np
import csv
import sys

# read data from file.
capital_max_g = 7000
capital_max_l = 390
trX_row = 32561
teX_row = 16281
t_col = 115
rawdata_trY = np.zeros((trX_row,1))
rawdata_teX = np.zeros((teX_row,t_col))

filename1 = sys.argv[3] # 'X_train'
filename2 = sys.argv[4] # 'Y_train'
filename3 = sys.argv[5] # 'X_test'
filename4 = sys.argv[6] # 'SUB_15'

file2 = open(filename2,'r+',encoding='Big5')
reader2 = csv.reader(file2)
for row in reader2:
	rawdata_trY[reader2.line_num-1,:] = row
file2.close()

c1_lin_recd = 0
c2_lin_recd = 0
c1_num = int(np.sum(rawdata_trY))
c2_num = trX_row-c1_num
rawdata_trX_c1 = np.zeros((c1_num,t_col))
rawdata_trX_c2 = np.zeros((c2_num,t_col))

file1 = open(filename1,'r+',encoding='Big5')
reader1 = csv.reader(file1)
for row in reader1:
	if(reader1.line_num==1):
		pass
	elif(rawdata_trY[reader1.line_num-2	,0]==1):
		rawdata_trX_c1[c1_lin_recd,:101] = [row[2]]+row[6:]
		rawdata_trX_c1[c1_lin_recd,101] = 1 if int(row[0])<=25 else 0
		rawdata_trX_c1[c1_lin_recd,102] = 1 if (int(row[0])>=26 and int(row[0])<=45) else 0
		rawdata_trX_c1[c1_lin_recd,103] = 1 if (int(row[0])>=46 and int(row[0])<=65) else 0
		rawdata_trX_c1[c1_lin_recd,104] = 1 if (int(row[0])>=66) else 0
		rawdata_trX_c1[c1_lin_recd,105] = 1 if int(row[5])<=25 else 0
		rawdata_trX_c1[c1_lin_recd,106] = 1 if (int(row[5])>=26 and int(row[5])<=40) else 0
		rawdata_trX_c1[c1_lin_recd,107] = 1 if (int(row[5])>=41 and int(row[5])<=60) else 0
		rawdata_trX_c1[c1_lin_recd,108] = 1 if (int(row[5])>=61) else 0
		rawdata_trX_c1[c1_lin_recd,109] = 1 if int(row[3])==0 else 0
		rawdata_trX_c1[c1_lin_recd,110] = 1 if (int(row[3])>0 and int(row[3])<capital_max_g) else 0
		rawdata_trX_c1[c1_lin_recd,111] = 1 if int(row[3])>=capital_max_g else 0
		rawdata_trX_c1[c1_lin_recd,112] = 1 if int(row[4])==0 else 0
		rawdata_trX_c1[c1_lin_recd,113] = 1 if (int(row[4])>0 and int(row[4])<capital_max_l) else 0
		rawdata_trX_c1[c1_lin_recd,114] = 1 if int(row[4])>=capital_max_l else 0
		c1_lin_recd += 1
	else:
		rawdata_trX_c2[c2_lin_recd,:101] = [row[2]]+row[6:]
		rawdata_trX_c2[c2_lin_recd,101] = 1 if int(row[0])<=25 else 0
		rawdata_trX_c2[c2_lin_recd,102] = 1 if (int(row[0])>=26 and int(row[0])<=45) else 0
		rawdata_trX_c2[c2_lin_recd,103] = 1 if (int(row[0])>=46 and int(row[0])<=65) else 0
		rawdata_trX_c2[c2_lin_recd,104] = 1 if (int(row[0])>=66) else 0
		rawdata_trX_c2[c2_lin_recd,105] = 1 if int(row[5])<=25 else 0
		rawdata_trX_c2[c2_lin_recd,106] = 1 if (int(row[5])>=26 and int(row[5])<=40) else 0
		rawdata_trX_c2[c2_lin_recd,107] = 1 if (int(row[5])>=41 and int(row[5])<=60) else 0
		rawdata_trX_c2[c2_lin_recd,108] = 1 if (int(row[5])>=61) else 0
		rawdata_trX_c2[c2_lin_recd,109] = 1 if int(row[3])==0 else 0
		rawdata_trX_c2[c2_lin_recd,110] = 1 if (int(row[3])>0 and int(row[3])<capital_max_g) else 0
		rawdata_trX_c2[c2_lin_recd,111] = 1 if int(row[3])>=capital_max_g else 0
		rawdata_trX_c2[c2_lin_recd,112] = 1 if int(row[4])==0 else 0
		rawdata_trX_c2[c2_lin_recd,113] = 1 if (int(row[4])>0 and int(row[4])<capital_max_l) else 0
		rawdata_trX_c2[c2_lin_recd,114] = 1 if int(row[4])>=capital_max_l else 0
		c2_lin_recd += 1
file1.close()

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
rawdata_trX_c1_sum = np.sum(rawdata_trX_c1,0)
rawdata_trX_c1_mean = rawdata_trX_c1_sum/c1_num
rawdata_trX_c1_std = np.std(rawdata_trX_c1,0)
rawdata_trX_c1_std = np.array([0.00001 if i==0 else i for i in rawdata_trX_c1_std])
rawdata_trX_c1 = (rawdata_trX_c1-rawdata_trX_c1_mean)/rawdata_trX_c1_std

rawdata_trX_c2_sum = np.sum(rawdata_trX_c2,0)
rawdata_trX_c2_mean = rawdata_trX_c2_sum/c2_num
rawdata_trX_c2_std = np.std(rawdata_trX_c2,0)
rawdata_trX_c2_std = np.array([0.00001 if i==0 else i for i in rawdata_trX_c2_std])
rawdata_trX_c2 = (rawdata_trX_c2-rawdata_trX_c2_mean)/rawdata_trX_c2_std

rawdata_teX_sum = np.sum(rawdata_teX,0)
rawdata_teX_mean = rawdata_teX_sum/teX_row
rawdata_teX_std = np.std(rawdata_teX,0)
rawdata_teX_std = np.array([0.00001 if i==0 else i for i in rawdata_teX_std])
rawdata_teX = (rawdata_teX-rawdata_teX_mean)/rawdata_teX_std


# mean & covariance
c1_mean = np.sum(rawdata_trX_c1,0)/c1_num
c1_cov = np.zeros((t_col,t_col))
for it in range(c1_num):
	c1_cov += (rawdata_trX_c1[it,:]-c1_mean).reshape(t_col,1).dot((rawdata_trX_c1[it,:]-c1_mean).reshape(1,t_col))
c2_mean = np.sum(rawdata_trX_c2,0)/(c2_num)
c2_cov = np.zeros((t_col,t_col))
for it in range(c2_num):
	c2_cov += (rawdata_trX_c2[it,:]-c2_mean).reshape(t_col,1).dot((rawdata_trX_c2[it,:]-c2_mean).reshape(1,t_col))
c1_postprob = c1_num/trX_row
c2_postprob = c2_num/trX_row
cov = c1_postprob*c1_cov+c2_postprob*c2_cov

# factors
w = (c1_mean-c2_mean).reshape(1,t_col).dot(np.linalg.inv(cov))
b = -0.5*c1_mean.reshape(1,t_col).dot(np.linalg.inv(cov).dot(c1_mean.reshape(t_col,1)))+0.5*c2_mean.reshape(1,t_col).dot(np.linalg.inv(cov).dot(c2_mean.reshape(t_col,1)))+np.log(c1_num/c2_num)


# Prediction for test set
prob_is_cl = 1/(1+np.exp(-(rawdata_teX.dot(w.T)+b[0,0]*np.ones((teX_row,1)))))
prob_is_cl = [1 if i>=0.5 else 0 for i in prob_is_cl]

file4 = open(filename4,'w+',encoding='Big5')
writer4 = csv.writer(file4)
writer4.writerow(['id','label'])
for it in range(teX_row):
	writer4.writerow([str(it+1),prob_is_cl[it]])
file4.close()


