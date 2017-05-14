#!/usr/bin/python3.6
#-*-coding:UTF-8-*-

import os
import sys
import glob
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc 
from scipy import linalg
import PIL
import PIL.Image as img

class PCA:
	"""docstring for PCA"""
	def __init__(self,pathname,count_num,char_count_num):
		trainData = []
		idx = "A"
		count = 0
		char_count = 1
		for file in sorted(os.listdir(pathname)):
			if file.endswith(".bmp"):
				if (file.startswith(idx) and (count < count_num)):
					filepath = pathname + file
					imgArr = np.asarray(img.open(filepath))
					imgList = np.reshape(imgArr, (np.product(imgArr.shape), )).astype('int')
					trainData.append(imgList)
					count += 1
				elif ((char_count < char_count_num) and (count == count_num)):
					count = 0
					char_count += 1
					idx = chr(ord(idx)+1)
		trainData = np.asarray(trainData)
		self.trainData = trainData

	def setCenter(self):
		dataMean = self.dataMean
		trainData = self.trainData
		for idx in np.arange(len(self.column(trainData,0))):
			trainData[idx] = trainData[idx] - dataMean
		dataAdjust = np.asarray(trainData)
		self.dataAdjust = dataAdjust
		return dataAdjust

	def SVD(self,matrix):
		[u,s,v] = linalg.svd(matrix)
		self.u = u
		self.s = s
		self.v = v
		return u,s,v

	def reduceDim(self,topEigenNum):
		s_red = []
		s = self.s
		for idx in np.arange(len(s)):
			if (idx < topEigenNum ):
				s_red.append(s[idx])
			else:
				s_red.append(0)
		s_red = np.asarray(s_red)
		S_red = linalg.diagsvd(s_red, 4096, 100)
		return S_red

	def column(self,matrix, i):
		return [row[i] for row in matrix]
	
	def findMean(self):
		dataMean =[]
		trainData = self.trainData
		for idx in np.arange(len(trainData[0])):
			meanValue = np.mean(self.column(trainData,idx))
			dataMean.append(meanValue)
		dataMean = np.asarray(dataMean)
		self.dataMean = dataMean
		return dataMean

	def reconData(self,u,s,v,mean):
		reconSet =[]
		recon = np.dot(np.dot(u,s),v)
		recon_t = recon.T
		for idx in np.arange(len(recon_t)):
			reconSet.append(recon_t[idx]+mean)
		reconSet = np.asarray(reconSet)
		return reconSet

	def uncenterData(self,matrix,mean):
		reconSet =[]
		for idx in np.arange(len(matrix)):
			reconSet.append(matrix[idx]+mean)
		reconSet = np.asarray(reconSet)
		return reconSet

	def averageFace(self,meanList):
		averageFace =np.asarray(meanList).reshape(64,64)
		plt.imshow(averageFace, cmap='gray')
		plt.savefig("average_face.jpg")
		#plt.show()

	def findEigenFace(self,u,s,vectorNum,mean):
		eigenFace = []
		temp = np.dot(u,s)
		temp = temp.T
		for idx in np.arange(vectorNum):
			eigenFace.append(temp[idx]+mean)
		eigenFace = np.asarray(eigenFace)
		return eigenFace

	def saveImg(self,inputArr,imgNum,whichone):
		num = int(np.sqrt(imgNum))
		imgArr = np.asarray(inputArr).reshape(len(inputArr),64,64)
		for idx in np.arange(imgNum):
			plt.subplot(num,num,idx+1)
			plt.imshow(imgArr[idx],cmap='gray')
			plt.axis('off')
		if (whichone=='top_9_eigenfaces'):
			plt.savefig("top_9_eigenfaces.jpg")
		if (whichone=='origin_faces'):
			plt.savefig("origin_faces.jpg")
		if (whichone=='recovered_faces'):
			plt.savefig("recovered_faces.jpg")

		#plt.show()



	
# main function.

pca_p1 = PCA(sys.argv[1],10,10)

meanList = pca_p1.findMean()

# Plot the average face
pca_p1.averageFace(meanList)

# plot the origin faces
pca_p1.saveImg(pca_p1.trainData,100,'origin_faces')

dataAdjust = pca_p1.setCenter()
data = pca_p1.uncenterData(dataAdjust,meanList)
dataAdjust_t = dataAdjust.T
u,s,v = pca_p1.SVD(dataAdjust_t)
s_red = pca_p1.reduceDim(9)
eigenFace = pca_p1.findEigenFace(u,s_red,9,meanList)

# Plot the eigen faces
pca_p1.saveImg(eigenFace,9,'top_9_eigenfaces')

s_red = pca_p1.reduceDim(5)
eigenFace = pca_p1.findEigenFace(u,s_red,5,meanList)
recon = pca_p1.reconData(u,s_red,v,meanList)

# Plot the recovered faces
pca_p1.saveImg(recon,100,'recovered_faces')

for k in range(200):
	s_red = pca_p1.reduceDim(k)
	eigenFace = pca_p1.findEigenFace(u,s_red,k,meanList)
	recon = pca_p1.reconData(u,s_red,v,meanList)
	error = np.sqrt(((data-recon)**2).mean())
	if((error/256)<0.01):
		print("The smallest k is",k ,", and the error is", error/256)
		break
