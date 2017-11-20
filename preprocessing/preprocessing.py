import numpy
import csv
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

def preproc(fileName, targetFrequency=250, targetSeconds=1):
	
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
	
	# split the signal to targetLength
	numberOfParts = numberOfValues / targetFrequency / targetSeconds
	counter = 0
	for subarray in numpy.vsplit(rsc, 10):
		numpy.savetxt(fileName + str(counter) + '.csv', subarray, delimiter=',')
		counter = counter + 1
		
def getActualFrequency(fileName):
	
	info = open(fileName + '.info','r')
	text = info.read()
	start = 'Sampling frequency: '
	end = 'Hz'
	return int((text.split(start))[1].split(end)[0])

# can be used with a .mat file of any signal length, frequency and number of leadss
# output will be presented in several .csv files
preproc('100m')