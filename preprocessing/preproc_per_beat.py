# script requires .mat, .info and .txt file (show rr intervals as text) of phyisionet record 

import numpy
import csv
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

def preproc(fileName, targetFrequency=250):
	
	# load mat file
	input = sio.loadmat(fileName + '.mat')
	x = input['val']
	
	# transpose x to a column vector
	xt = numpy.transpose(x)
	
	# change frequency
	numberOfValues = (len(xt) / getActualFrequency(fileName)) * targetFrequency
	xnf = signal.resample(xt, numberOfValues)
				
	# rescale the signal
	mn = numpy.min(xnf)
	mx = numpy.max(xnf)
	rsc = (xnf - mn) / (mx - mn)
	
	# split the signal
	splitInfo = getSplitInfo(fileName)
	alreadUsedValues = 0
	counter = 0
	for subArray in splitInfo:
		numberOfValuesToUse = int(round(targetFrequency * float(subArray[0])))
		finalValues = rsc[alreadUsedValues:(alreadUsedValues+numberOfValuesToUse)]
		numpy.savetxt(fileName+'-'+str(counter)+'-'+subArray[1]+'.csv', finalValues, fmt='%0.2f', delimiter=',')
		alreadUsedValues += numberOfValuesToUse
		counter += 1

def getActualFrequency(fileName):
	
	info = open(fileName + '.info','r')
	text = info.read()
	start = 'Sampling frequency: '
	end = 'Hz'
	return int((text.split(start))[1].split(end)[0])
	
def getSplitInfo(fileName):
	
	info = open(fileName + '.txt','r')
	splitInfo = []
	for line in info:
		y = line.split()
		splitInfo.append([y[2], y[3]])
	return splitInfo

# can be used with a .mat file of any signal length, frequency and number of leads
# output will be presented in several .csv files
# label is in generated csv file name
preproc('100m')